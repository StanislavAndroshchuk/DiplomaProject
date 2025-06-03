import math
import random
from .maze import Maze, CELL_WALL

class Agent:
    """
    Клас, що представляє агента, керованого нейромережею,
    який рухається в лабіринті.
    """

    def __init__(self, agent_id: int, start_pos: tuple[int, int], config: dict):
        self.id = agent_id
        self.x = float(start_pos[1]) + 0.5
        self.y = float(start_pos[0]) + 0.5
        self.angle = random.uniform(0, 2 * math.pi)
        self.velocity = 0.0
        self.max_speed = config.get('agent_max_speed', config.get('AGENT_MAX_SPEED', 0.5)) # Додав fallback до config
        self.radius = 0.3
        self.num_rangefinders = config['NUM_RANGEFINDERS']
        self.rangefinder_angles_relative = [i * (2 * math.pi / self.num_rangefinders) for i in range(self.num_rangefinders)]
        self.rangefinder_max_dist = config['RANGEFINDER_MAX_DIST']
        self.num_radar_slices = config['NUM_RADAR_SLICES']
        self.radar_slice_angle = 2 * math.pi / self.num_radar_slices
        self.rangefinder_readings = [1.0] * self.num_rangefinders
        self.radar_readings = [0.0] * self.num_radar_slices
        self.heading_vector = (math.cos(self.angle), math.sin(self.angle)) # Оновлено на основі кута
        self.current_velocity_reading = 0.0
        self.last_rangefinder_rays = []
        self.steps_taken = 0
        self.collided = False
        self.reached_goal = False
        self.min_dist_to_goal = float('inf')

    def _cast_ray(self, maze: Maze, angle_offset: float, max_dist: float) -> tuple[float, float, float, float, float]:
        ray_angle_global = self.angle + angle_offset 
        cos_a = math.cos(ray_angle_global)
        sin_a = math.sin(ray_angle_global)
        # Крок перевірки можна зробити меншим для більшої точності, але це вплине на продуктивність
        step_size = 0.1
        start_x_ray, start_y_ray = self.x, self.y
        current_dist = 0.0
        # Кінцева точка променя, яку ми будемо оновлювати
        ray_end_x, ray_end_y = start_x_ray, start_y_ray 

        while current_dist < max_dist:
            check_x = start_x_ray + cos_a * current_dist
            check_y = start_y_ray + sin_a * current_dist
            map_r, map_c = int(check_y), int(check_x)
            if not maze._is_valid(map_r, map_c):
                ray_end_x = start_x_ray + cos_a * current_dist # Точка на промені
                ray_end_y = start_y_ray + sin_a * current_dist
                return start_x_ray, start_y_ray, ray_end_x, ray_end_y, current_dist
            if maze.grid[map_r][map_c] == CELL_WALL:
                ray_end_x = start_x_ray + cos_a * current_dist 
                ray_end_y = start_y_ray + sin_a * current_dist
                return start_x_ray, start_y_ray, ray_end_x, ray_end_y, current_dist
            current_dist += step_size

        # Якщо стіна не знайдена в межах max_dist
        ray_end_x = start_x_ray + cos_a * max_dist
        ray_end_y = start_y_ray + sin_a * max_dist
        return start_x_ray, start_y_ray, ray_end_x, ray_end_y, max_dist

    def get_sensor_readings(self, maze: Maze) -> list[float]:
        self.last_rangefinder_rays = [] 

        # 1. Датчики відстані (Rangefinders)
        for i, angle_offset in enumerate(self.rangefinder_angles_relative):
            start_x, start_y, end_x, end_y, actual_dist = self._cast_ray(maze, angle_offset, self.rangefinder_max_dist)
            # нормована відстань для нейромережі
            self.rangefinder_readings[i] = actual_dist / self.rangefinder_max_dist
            # дані про промінь для візуалізації
            self.last_rangefinder_rays.append((start_x, start_y, end_x, end_y, actual_dist))

        # 2. Радар до цілі (Goal Radar)
        goal_pos = maze.goal_pos
        if goal_pos:
             goal_center_x = float(goal_pos[1]) + 0.5
             goal_center_y = float(goal_pos[0]) + 0.5
             dx_goal = goal_center_x - self.x
             dy_goal = goal_center_y - self.y
             angle_to_goal_global = math.atan2(dy_goal, dx_goal)
             relative_angle = (angle_to_goal_global - self.angle + math.pi) % (2 * math.pi) - math.pi
             self.radar_readings = [0.0] * self.num_radar_slices
             # Визначаємо, в який сектор потрапляє кут до цілі відносно напрямку агента
             # Перетворюємо relative_angle в діапазон [0, 2*pi) для простого розрахунку індексу
             positive_relative_angle = (relative_angle + 2 * math.pi) % (2 * math.pi)
             target_sector_index = int(positive_relative_angle / self.radar_slice_angle)
             # Перевірка, чи індекс не виходить за межі (мало б бути добре через %)
             target_sector_index = min(target_sector_index, self.num_radar_slices - 1)
             if 0 <= target_sector_index < self.num_radar_slices:
                 self.radar_readings[target_sector_index] = 1.0
        else:
             self.radar_readings = [0.0] * self.num_radar_slices

        self.heading_vector = (math.cos(self.angle), math.sin(self.angle))
        self.current_velocity_reading = self.velocity / self.max_speed if self.max_speed != 0 else 0.0
        sensor_data = []
        sensor_data.extend(self.rangefinder_readings)
        sensor_data.extend(self.radar_readings)
        sensor_data.extend(list(self.heading_vector))
        sensor_data.append(self.current_velocity_reading)

        return sensor_data

    def update(self, maze: Maze, network_outputs: list[float], dt: float = 0.5):
        """
        Оновлює стан агента (кут, швидкість, позиція) на основі виходів нейромережі.
        Враховує просту фізику та колізії зі стінами.
        Приймає 4 виходи: [TurnL, TurnR, Accel, Brake].

        Args:
            maze (Maze): Об'єкт лабіринту для перевірки колізій.
            network_outputs (list[float]): Список з 4 вихідних значень з нейромережі.
                                           Порядок: [TurnL, TurnR, Accel, Brake]
            dt (float): Часовий крок симуляції (для фізики).
        """
        expected_outputs = 4
        if len(network_outputs) != expected_outputs: 
             print(f"Warning: Agent {self.id} received {len(network_outputs)} outputs, expected {expected_outputs}.")
             network_outputs = [0.5] * expected_outputs # Нейтральні значення

        turn_left_signal = network_outputs[0]
        turn_right_signal = network_outputs[1]
        accel_signal = network_outputs[2]
        brake_signal = network_outputs[3]

        # 1. Розрахунок зміни кута
        max_turn_rate = math.pi / 2 * dt # 90 градусів за 1 сек
        turn_request = 0.0
        # Просте віднімання сигналів. 0.5 = нейтрально. > 0.5 = поворот.
        turn_strength_left = max(0.0, turn_left_signal - 0.5) * 2
        turn_strength_right = max(0.0, turn_right_signal - 0.5) * 2
        turn_request = (turn_strength_right - turn_strength_left) * max_turn_rate
        self.angle = (self.angle + turn_request) % (2 * math.pi)

        # 2. Розрахунок зміни швидкості
        accel_power = 0.2 * self.max_speed * dt # Макс. зміна швидкості за крок
        brake_power = 0.4 * self.max_speed * dt # Сила гальмування
        friction = 0.05 * dt # Коефіцієнт тертя
        # Прискорення, якщо сигнал > 0.5
        acceleration = max(0.0, accel_signal - 0.5) * 2 * accel_power
        # Гальмування, якщо сигнал > 0.5
        braking = max(0.0, brake_signal - 0.5) * 2 * brake_power

        self.velocity += acceleration
        self.velocity -= braking
        self.velocity *= (1.0 - friction) # тертя
        self.velocity = max(0.0, min(self.max_speed, self.velocity))

        # 3. Розрахунок переміщення
        move_dist = self.velocity * dt
        dx = math.cos(self.angle) * move_dist
        dy = math.sin(self.angle) * move_dist
        new_x = self.x + dx
        new_y = self.y + dy

        # 4. Перевірка колізій
        target_r, target_c = int(new_y), int(new_x)
        self.collided = False
        if not maze.is_walkable(target_r, target_c):
            self.velocity = 0 # Зупинка при зіткненні
            self.collided = True
        else:
            self.x = new_x
            self.y = new_y

        # 5. Оновлення стану для оцінки
        goal_pos = maze.goal_pos
        if goal_pos:
            dist = math.hypot(self.x - (goal_pos[1] + 0.5), self.y - (goal_pos[0] + 0.5))
            self.min_dist_to_goal = min(self.min_dist_to_goal, dist)
            current_r, current_c = self.get_position_int()
            if current_r == goal_pos[0] and current_c == goal_pos[1]:
                self.reached_goal = True

    def get_position_int(self) -> tuple[int, int]:
         """Повертає цілочисельні координати (row, col) агента."""
         return (int(self.y), int(self.x))

    def reset(self, start_pos: tuple[int, int]):
        """Скидає стан агента до початкового."""
        self.x = float(start_pos[1]) + 0.5
        self.y = float(start_pos[0]) + 0.5
        self.angle = random.uniform(0, 2 * math.pi)
        self.velocity = 0.0
        self.rangefinder_readings = [1.0] * self.num_rangefinders
        self.radar_readings = [0.0] * self.num_radar_slices
        self.heading_vector = (math.cos(self.angle), math.sin(self.angle)) 
        self.current_velocity_reading = 0.0
        self.steps_taken = 0
        self.collided = False
        self.reached_goal = False
        self.min_dist_to_goal = float('inf')