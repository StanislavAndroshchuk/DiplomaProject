import os
import tkinter as tk
import time
import pickle
import math
import importlib
from tkinter import filedialog,messagebox
from tkinter import ttk
from typing import Optional, Tuple

from neat.json_serializer import NEATJSONSerializer

try:
    import config as cfg
except ImportError:
    print("ERROR: config.py not found. Make sure it's in the project root.")
    exit()

from environment.maze import Maze
from environment.agent import Agent
from neat.neat_algorithm import NeatAlgorithm
from neat.genome import Genome
from neat.nn import activate_network
from visualization.gui import MazeGUI


def evaluate_single_genome(genome_tuple: Tuple[int, Genome], config: dict) -> Tuple[int, float, bool]:
    """
    Оцінює ОДИН геном. Приймає кортеж (id, genome) та конфіг.
    Повертає кортеж (id, fitness, reached_goal_flag).
    """
    genome_id, genome = genome_tuple
    if not genome:
        return genome_id, 0.001, False

    try:
        maze = Maze(config['MAZE_WIDTH'], config['MAZE_HEIGHT'], config.get('MAZE_SEED'))
        if not maze.start_pos:
             print(f"Error (process): Maze generation failed for genome {genome_id}")
             return genome_id, 0.001, False

        agent = Agent(genome_id, maze.start_pos, config.copy())
        max_steps = config.get('MAX_STEPS_PER_EVALUATION', 500)
        
        genome_reached_goal_flag = False 
        for step in range(max_steps):
            if agent.reached_goal:
                genome_reached_goal_flag = True 
                break 
            sensor_readings = agent.get_sensor_readings(maze)
            network_outputs = activate_network(genome.copy(), sensor_readings)
            if network_outputs is None: 
                 print(f"Error (process): activate_network failed for genome {genome_id}")
                 return genome_id, 0.001, False
            agent.update(maze, network_outputs, dt=1)
            agent.steps_taken = step + 1
        
        # Перевіряємо ще раз після циклу, якщо ціль досягнута на останньому кроці
        if agent.reached_goal and not genome_reached_goal_flag:
            genome_reached_goal_flag = True

        IS_REWARDING_EFFICIENCY = config['IS_REWARDING_EFFICIENCY'] if 'IS_REWARDING_EFFICIENCY' in config else True
        fitness = 0.0
        BASE_REWARD = 1000.0
        EFFICIENCY_REWARD = 1000.0
        # Агенти починають рух з центру клітинки старту, тому додаємо 0.5
        x_start_pos, y_start_pos = float(maze.start_pos[0]+0.5), float(maze.start_pos[1]+0.5)
        # Обрахунок максимальної відстані від старту до цілі 
        # (від позиції (1.5;1.5) до (10.5;10.5) для лабіринту 11x11)
        bf = math.hypot(maze.goal_pos[0] + 0.5 - x_start_pos, maze.goal_pos[1] + 0.5 - y_start_pos)
        # Мінімальна відстань до цілі, яку агент досяг
        dg = agent.min_dist_to_goal
        # Різниця між максимальною та мінімальною відстанню
        fitness = bf - dg 
        if agent.reached_goal:
            fitness = BASE_REWARD
            if max_steps > 0 and IS_REWARDING_EFFICIENCY:
                # Нормалізований коефіцієнт ефективності: 1.0 (найкраще) -> 0.0 (найгірше)
                efficiency_ratio = 1.0 - (float(agent.steps_taken) / max_steps)
                fitness += EFFICIENCY_REWARD * max(0.0, efficiency_ratio)
        else:
            if fitness < 0: # Якщо агент не рухається до цілі
                fitness = 0.001
            else:
                proximity = fitness / bf 
                fitness = max(0.001, proximity) * BASE_REWARD
                if fitness > BASE_REWARD:
                    raise ValueError("Фітнес не може бути більшим за базову винагороду!")
        
        return genome_id, max(0.001, fitness), genome_reached_goal_flag

    except Exception as e:
        print(f"Error evaluating genome {genome_id} in parallel process: {e}")
        import traceback
        traceback.print_exc()
        return genome_id, 0.001, False 

class SimulationController:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.config = self._load_config()
        self.config['NUM_PROCESSES'] = os.cpu_count()
        print(f"Using {self.config['NUM_PROCESSES']} processes for evaluation.")

        try:
            num_inputs = (
                self.config['NUM_RANGEFINDERS'] +
                self.config['NUM_RADAR_SLICES'] +
                2 +  # heading_vector (x, y)
                1    # current_velocity_reading
            )
            num_outputs = self.config['NUM_OUTPUTS']
            print(f"Network configuration: {num_inputs} Inputs, {num_outputs} Outputs") 
        except KeyError as e:
             print(f"FATAL ERROR: Missing configuration key needed for NN setup: {e}")
             master.quit()
             return

        self.config['NUM_INPUTS'] = num_inputs
        self.neat = NeatAlgorithm(self.config, num_inputs, num_outputs, initial_genome_id_start=0)
        
        self.maze = Maze(self.config['MAZE_WIDTH'], self.config['MAZE_HEIGHT'], self.config.get('MAZE_SEED'))
        self.config['MAZE_SEED'] = self.maze.seed 

        self.agents = {} 

        self.gui = MazeGUI(master, self.config, self)

        self._is_running_multiple = False
        self._stop_multiple_requested = False
        self.is_running = False
        self.current_simulation_step = 0
        self.max_steps_per_gen_vis = self.config.get('MAX_STEPS_PER_EVALUATION', 500)

        self._redraw_maze()
        self._update_gui_stats()

    def _load_config(self) -> dict:
        """Завантажує конфігурацію з config.py."""
        try:
            importlib.reload(cfg)
            config_dict = {key: getattr(cfg, key) for key in dir(cfg) if not key.startswith('_')}
        except Exception as e:
             print(f"ERROR loading config.py: {e}")
             return {}

        config_dict.setdefault('POPULATION_SIZE', 150)
        config_dict.setdefault('MAZE_WIDTH', 21)
        config_dict.setdefault('MAZE_HEIGHT', 15)
        config_dict.setdefault('NUM_RANGEFINDERS', 8)
        config_dict.setdefault('NUM_RADAR_SLICES', 8)
        config_dict.setdefault('NUM_OUTPUTS', 7) 
        return config_dict


    def _redraw_maze(self):
        """Перемальовує лабіринт на GUI."""
        self.gui.draw_maze(self.maze.grid, self.config.get('CELL_SIZE_PX', 20))

    def _reset_agents_for_visualization(self):
        """Скидає агентів для візуалізації на основі поточної популяції NEAT."""
        self.agents.clear()
        self.gui.clear_all_agents()
        pop_size = len(self.neat.population) if self.neat and self.neat.population else 0
        print(f"Resetting agents for visualization. Population size: {pop_size}")
        if not self.neat.population: return

        for genome in self.neat.population:
             if self.maze.start_pos:
                 if genome:
                     self.agents[genome.id] = Agent(genome.id, self.maze.start_pos, self.config)
                 else:
                     print("Warning: Found None genome while resetting agents.")
             else:
                  print("Error: Cannot reset agents, maze has no start position.")
        print(f"Agents created for visualization: {len(self.agents)}")

    def _update_agents_visuals(self):
         """Оновлює позиції всіх агентів на GUI."""
         best_gen_genome_id = None
         if self.neat.population:
             current_best_gen = max(self.neat.population, key=lambda g: g.fitness if g else -float('inf'), default=None) # Додав перевірку на None
             if current_best_gen: best_gen_genome_id = current_best_gen.id

         best_overall_genome_id = self.neat.best_genome_overall.id if self.neat.best_genome_overall else None

         self.gui.clear_all_agents() 
         
         show_sensors_flag = self.gui.show_sensors_var.get() if hasattr(self.gui, 'show_sensors_var') else False

         agents_to_draw = list(self.agents.keys())
         for agent_id in agents_to_draw:
             agent = self.agents.get(agent_id)
             if agent:
                 is_best_gen = (agent_id == best_gen_genome_id)
                 is_best_overall = (agent_id == best_overall_genome_id)
                 self.gui.update_agent_visual(
                     agent.id, agent.x, agent.y, agent.angle,
                     is_best_gen=is_best_gen, is_best_overall=is_best_overall,
                     show_sensors=show_sensors_flag 
                 )
    def get_genome_by_id(self, genome_id: int) -> Optional[Genome]:
        """Шукає геном за ID у поточній популяції."""
        if not self.neat or not self.neat.population:
            return None
        
        try:
            target_id = int(genome_id)
        except ValueError:
            print(f"Warning: Invalid genome_id format passed to get_genome_by_id: {genome_id}")
            return None

        for genome in self.neat.population:
            if genome and genome.id == target_id: 
                return genome
        
        if self.neat.best_genome_overall and self.neat.best_genome_overall.id == target_id:
            return self.neat.best_genome_overall
        
        if self.neat.species:
            for spec in self.neat.species:
                for member in spec.members:
                    if member and member.id == target_id:
                        print(f"Info: Genome {target_id} found in species {spec.id} members, but not in main population or best_overall.")
                        return member
                if spec.representative and spec.representative.id == target_id:
                    print(f"Info: Genome {target_id} found as representative of species {spec.id}, but not in main population or best_overall.")
                    return spec.representative
        return None 
    
    def _update_gui_stats(self):
         """Отримує статистику та передає в GUI для безпечного оновлення."""
         if not hasattr(self, 'neat'): return 

         stats = {
             "generation": self.neat.generation,
             "num_species": len(self.neat.species),
             "max_fitness": None,
             "average_fitness": None,
             "best_overall_fitness": None,
             "best_genome_current_gen": None,
             "best_genome_overall": self.neat.best_genome_overall,
             "first_goal_achieved_generation": self.neat.first_goal_achieved_generation 
         }
         current_best_genome = None
         if self.neat.population:
             valid_genomes = [g for g in self.neat.population if g is not None]
             if valid_genomes:
                  current_best_genome = max(valid_genomes, key=lambda g: g.fitness, default=None)
                  if current_best_genome:
                       stats["max_fitness"] = current_best_genome.fitness
                  total_fitness = sum(g.fitness for g in valid_genomes)
                  stats["average_fitness"] = total_fitness / len(valid_genomes) if valid_genomes else 0.0
                  stats["best_genome_current_gen"] = current_best_genome

         if self.neat.best_genome_overall:
             stats["best_overall_fitness"] = self.neat.best_genome_overall.fitness

         # Викликаємо безпечне оновлення GUI
         self.gui.update_gui_from_thread(stats)

    def toggle_simulation(self, run: bool):
        """Запускає або ставить на паузу ВІЗУАЛІЗАЦІЮ."""
        if self._is_running_multiple:
            print("Cannot start/pause visualization while batch run is active.")
            self.gui.set_controls_state(running=False, running_multiple=True)
            # І що обидві візуалізації точно зупинені
            self.gui.is_running = False
            if hasattr(self.gui, 'is_best_agent_viz_running'):
                self.gui.is_best_agent_viz_running = False
            return


        if run: 
            self.current_simulation_step = 0
            if not self.agents: 
                self._reset_agents_for_visualization()
            
            if self.agents: 
                 self.master.after(50, self.simulation_step)
            else: 
                 print("Controller: No agents to visualize for main population.")
                 self.gui.is_running = False 
                 if hasattr(self.gui, 'is_best_agent_viz_running'):
                     self.gui.is_best_agent_viz_running = False 
                 self.gui.set_controls_state(running=False, running_multiple=self._is_running_multiple)
        else: 
            if hasattr(self.gui, 'is_best_agent_viz_running'):
                self.gui.is_best_agent_viz_running = False
            self.gui.set_controls_state(running=False, running_multiple=self._is_running_multiple)
                 
    def simulation_step(self):
        """Виконує один крок симуляції для візуалізації."""
        if not self.gui.is_running or self._is_running_multiple:
            if hasattr(self.gui, 'is_best_agent_viz_running'):
                self.gui.is_best_agent_viz_running = False
            
            self.gui.set_controls_state(running=False, running_multiple=self._is_running_multiple)
            return

        start_time = time.time()

        active_agents = 0
        genomes_map = {genome.id: genome for genome in self.neat.population if genome}
        agents_to_remove = [] 

        agents_alive_this_step = list(self.agents.items()) 

        for agent_id, agent in agents_alive_this_step:
            if agent_id not in self.agents: continue

            if agent.reached_goal or self.current_simulation_step >= self.max_steps_per_gen_vis:
                agents_to_remove.append(agent_id)
                continue

            active_agents += 1
            genome = genomes_map.get(agent_id)
            if not genome:
                agents_to_remove.append(agent_id)
                continue

            # Отримуємо сенсорні дані
            sensor_readings = agent.get_sensor_readings(self.maze) 
            if not isinstance(sensor_readings, list): 
                print(f"ERROR: agent.get_sensor_readings for agent {agent_id} returned {type(sensor_readings)}, expected list.")
                agents_to_remove.append(agent_id)
                continue

            # --- Блок ДІАГНОСТИКИ ---
            network_outputs = None
            try:
                network_outputs = activate_network(genome, sensor_readings)

                # !!! Перевіряємо результат ПЕРЕД викликом update !!!
                if network_outputs is None:
                    print(f"FATAL ERROR: activate_network for genome {genome.id} returned None!")
                    raise TypeError("activate_network returned None unexpectedly.")
                if not isinstance(network_outputs, list):
                     print(f"FATAL ERROR: activate_network for genome {genome.id} returned {type(network_outputs)}, expected list!")
                     raise TypeError("activate_network did not return a list.")
                agent.update(self.maze, network_outputs, dt=1)
            except Exception as e:
                print(f"Error updating agent {agent_id} with genome {genome.id}: {e}")
                agents_to_remove.append(agent_id)

        for agent_id in agents_to_remove:
             if agent_id in self.agents:
                 del self.agents[agent_id]

        self._update_agents_visuals()
        self.gui.update_gui() 

        self.current_simulation_step += 1

        # Перевірка умов зупинки візуалізації (ОСНОВНОЇ ПОПУЛЯЦІЇ)
        if not self.agents or self.current_simulation_step >= self.max_steps_per_gen_vis:
             # print(f"Main population visualization step limit reached ({self.current_simulation_step}) or no active agents left.")
             self.gui.is_running = False 
             
             if hasattr(self.gui, 'is_best_agent_viz_running'):
                self.gui.is_best_agent_viz_running = False
             
             self.gui.set_controls_state(running=False, running_multiple=self._is_running_multiple)
             return 

        elapsed = time.time() - start_time
        delay = max(1, 30 - int(elapsed * 1000))
        self.master.after(delay, self.simulation_step)


    def run_one_generation(self):
        """Запускає ОДИН повний цикл покоління NEAT з паралельною оцінкою."""
        if self._is_running_multiple: 
            print("Cannot run single generation while multiple generations are running.")
            return

        print(f"\n--- Running Generation {self.neat.generation + 1} ---")
        start_time = time.time()
        stats = self.neat.run_generation(evaluate_single_genome)

        end_time = time.time()
        gen_num = stats.get('generation', '?')
        max_fit = stats.get('max_fitness', float('nan'))
        avg_fit = stats.get('average_fitness', float('nan'))
        num_sp = stats.get('num_species', '?')

        print(f"Generation {gen_num} finished in {end_time - start_time:.2f} seconds.")
        print(f"Stats: MaxFit={max_fit:.4f}, AvgFit={avg_fit:.4f}, Species={num_sp}")

        gui_stats_payload = {
            "generation": stats.get('generation'),
            "num_species": stats.get('num_species_after_speciation', len(self.neat.species)),
            "max_fitness": stats.get('max_fitness'),
            "average_fitness": stats.get('average_fitness'),
            "best_genome_current_gen": stats.get('best_genome_current_gen'),
            "best_genome_overall": self.neat.best_genome_overall,
            "best_overall_fitness": self.neat.best_genome_overall.fitness if self.neat.best_genome_overall else None,
            "first_goal_achieved_generation": self.neat.first_goal_achieved_generation
        }
        self.gui.update_gui_from_thread(gui_stats_payload)

        self._reset_agents_for_visualization()
        self._update_agents_visuals()
        self.gui.update_gui()

    def run_multiple_generations(self, num_generations: int):
        """Запускає вказану кількість поколінь NEAT у фоновому потоці."""
        print(f"Starting batch run for {num_generations} generations...")
        self._is_running_multiple = True
        self._stop_multiple_requested = False

        start_gen = self.neat.generation + 1
        end_gen = start_gen + num_generations

        try:
            for gen in range(start_gen, end_gen):
                 if self._stop_multiple_requested:
                     print("Batch run interrupted by user.")
                     break

                 print(f"\n--- Running Generation {gen} (Batch) ---")
                 start_time_gen = time.time()

                 stats = self.neat.run_generation(evaluate_single_genome) 

                 end_time_gen = time.time()
                 max_fit = stats.get('max_fitness', float('nan'))
                 avg_fit = stats.get('average_fitness', float('nan'))
                 num_sp = stats.get('num_species', '?')

                 print(f"Generation {gen} finished in {end_time_gen - start_time_gen:.2f} sec. MaxFit={max_fit:.4f}, AvgFit={avg_fit:.4f}, Sp={num_sp}")

                 gui_stats_payload = {
                    "generation": stats.get('generation'),
                    "num_species": stats.get('num_species_after_speciation', len(self.neat.species)),
                    "max_fitness": stats.get('max_fitness'),
                    "average_fitness": stats.get('average_fitness'),
                    "best_genome_current_gen": stats.get('best_genome_current_gen'),
                    "best_genome_overall": self.neat.best_genome_overall, 
                    "best_overall_fitness": self.neat.best_genome_overall.fitness if self.neat.best_genome_overall else None,
                    "first_goal_achieved_generation": self.neat.first_goal_achieved_generation
                }
                 self.gui.update_gui_from_thread(gui_stats_payload) 


        except Exception as e:
            print(f"\n--- ERROR during batch run at generation {self.neat.generation} ---")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Batch Run Error", f"An error occurred:\n{e}")
        finally:
            print("Batch run finished.")
            self._is_running_multiple = False
            self._stop_multiple_requested = False
            self.master.after(0, self.gui.set_controls_state, False, False)
            self.master.after(0, self._reset_agents_for_visualization)
            self.master.after(0, self._update_agents_visuals)


    def generate_new_maze(self, seed=None) -> int | None:
        """Генерує новий лабіринт і оновлює GUI."""
        print(f"Generating new maze with seed: {seed}")
        self.config['MAZE_SEED'] = seed 
        try:
             w = self.config['MAZE_WIDTH']
             h = self.config['MAZE_HEIGHT']
             if w % 2 == 0: w += 1; print(f"Adjusted MAZE_WIDTH to {w} (must be odd)")
             if h % 2 == 0: h += 1; print(f"Adjusted MAZE_HEIGHT to {h} (must be odd)")
             w = max(5, w); h = max(5, h)
             self.config['MAZE_WIDTH'] = w
             self.config['MAZE_HEIGHT'] = h
             
             self.maze = Maze(w, h, self.config['MAZE_SEED']) 
             self.config['MAZE_SEED'] = self.maze.seed 
             
             self._redraw_maze()
             self._reset_agents_for_visualization() 
             self._update_agents_visuals()
             self.gui.update_gui()
             return self.maze.seed 
        except ValueError as e:
             messagebox.showerror("Maze Generation Error", f"Failed to generate maze: {e}\nCheck MAZE_WIDTH/MAZE_HEIGHT in config (must be odd >= 5).")
             return self.config.get('MAZE_SEED') 

    def save_simulation(self):
        if self._is_running_multiple: 
            messagebox.showwarning("Saving Denied", "Cannot save state while multiple generations are running. Please wait for completion or stop the batch process.")
            return

        if not self.neat:
            messagebox.showerror("Error", "NEAT algorithm not initialized.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Save NEAT Simulation State"
        )
        if not filepath:
            return

        try:
            NEATJSONSerializer.save_neat_state(filepath, self.neat, self.config.copy())
            messagebox.showinfo("Success", f"Simulation state saved to {os.path.basename(filepath)}")
            print(f"State saved successfully to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save simulation state:\n{e}")
            print(f"Error saving simulation: {e}")
            import traceback
            traceback.print_exc()

    def load_simulation(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("NEAT Save Files (Legacy)", "*.neat_save"), ("All Files", "*.*")],
            title="Load NEAT Simulation State"
        )
        if not filepath:
            return

        try:
            if filepath.endswith('.json'):
                from neat.genome import Genome, NodeGene, ConnectionGene
                from neat.innovation import InnovationManager
                from neat.species import Species
                
                self.neat = NEATJSONSerializer.load_neat_state(
                    filepath,
                    self.config,
                    NeatAlgorithm,
                    Genome,
                    NodeGene,
                    ConnectionGene,
                    Species,
                    InnovationManager
                )
                
                maze_seed = self.config.get('MAZE_SEED')
                self.maze = Maze(self.config['MAZE_WIDTH'], self.config['MAZE_HEIGHT'], maze_seed)
                self._redraw_maze()
                
            else: 
                with open(filepath, 'rb') as f:
                    simulation_data = pickle.load(f)

                save_version = simulation_data.get('version', 'unknown')
                print(f"Loading save file version: {save_version}")

                loaded_config = simulation_data.get('config')
                if loaded_config:
                    num_processes_current = self.config.get('NUM_PROCESSES')
                    self.config.update(loaded_config) 
                    self.config['NUM_PROCESSES'] = num_processes_current
                else:
                    print("Warning: No config found in save file. Using current config.")

                maze_seed = self.config.get('MAZE_SEED')
                
                w = self.config.get('MAZE_WIDTH', 11)
                h = self.config.get('MAZE_HEIGHT', 11)
                if w % 2 == 0: w += 1
                if h % 2 == 0: h += 1
                w = max(5, w); h = max(5, h)
                self.config['MAZE_WIDTH'] = w
                self.config['MAZE_HEIGHT'] = h
                self.maze = Maze(w, h, maze_seed) 
                self.config['MAZE_SEED'] = self.maze.seed 
                self._redraw_maze()

                neat_state_data = simulation_data.get('neat_algorithm_state')
                num_inputs = self.config['NUM_INPUTS'] 
                num_outputs = self.config['NUM_OUTPUTS']

                if neat_state_data:
                    self.neat = NeatAlgorithm.load_from_state_data(neat_state_data, self.config, num_inputs, num_outputs)
                else:
                    messagebox.showerror("Load Error", "NEAT algorithm data not found in save file.")
                    return
                
            self._update_gui_stats()
            self._reset_agents_for_visualization()
            self._update_agents_visuals()
            self.gui.set_controls_state(running=False, running_multiple=False)
            self.gui.update_gui()

            messagebox.showinfo("Success", f"Simulation state loaded from {os.path.basename(filepath)}")
            print(f"Simulation loaded. Current generation: {self.neat.generation}, Maze Seed: {self.config.get('MAZE_SEED')}")

        except FileNotFoundError:
            messagebox.showerror("Load Error", f"Save file not found: {filepath}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load simulation state:\n{e}")
            print(f"Error loading simulation: {e}")
            import traceback
            traceback.print_exc()
    def reset_simulation(self):
        """Скидає симуляцію NEAT до початкового стану."""
        print("Resetting NEAT simulation...")
        try:
            self.config = self._load_config() 
            self.config['NUM_PROCESSES'] = os.cpu_count() 
            num_inputs = (self.config['NUM_RANGEFINDERS'] + self.config['NUM_RADAR_SLICES'] + 2 + 1)
            num_outputs = self.config['NUM_OUTPUTS']
            self.config['NUM_INPUTS'] = num_inputs

            self.neat = NeatAlgorithm(self.config, num_inputs, num_outputs) 

            self.generate_new_maze(self.config.get('MAZE_SEED')) 

            self._update_gui_stats() 
            self.gui.update_gui()
            print(f"Simulation reset complete. Maze Seed: {self.config.get('MAZE_SEED')}")
        except Exception as e:
            print(f"FATAL ERROR during reset: {e}")
            messagebox.showerror("Reset Error", f"Could not reset simulation: {e}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 

    root = tk.Tk()
    try:
        controller = SimulationController(root)
        root.mainloop()
    except Exception as e:
        print(f"\n--- Unhandled Exception ---")
        import traceback
        traceback.print_exc()
        print(f"---------------------------\n")
        try:
             messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nCheck console output.")
        except: pass