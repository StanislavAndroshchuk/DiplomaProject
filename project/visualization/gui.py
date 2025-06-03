import math
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, Menu
from PIL import Image, ImageTk
from typing import Optional, TYPE_CHECKING
import threading
if TYPE_CHECKING:
    from project.main import SimulationController 

try:
    from .network_visualizer import visualize_network
except ImportError:
    print("Warning: Could not import visualize_network from network_visualizer.")
    pass

try:
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from environment.maze import Maze, CELL_WALL, CELL_PATH, CELL_START, CELL_GOAL 
    from environment.agent import Agent 
    from neat.nn import activate_network 
except ImportError:
    print("Warning: Could not import constants from environment.maze. Using default values.")
    CELL_PATH, CELL_WALL, CELL_START, CELL_GOAL = 0, 1, 2, 3

# --- Константи для кольорів ---
COLOR_WALL = "black"
COLOR_PATH = "white"
COLOR_START = "lightgreen"
COLOR_GOAL = "red"
COLOR_AGENT_DEFAULT = "blue"
COLOR_AGENT_BEST = "yellow"
COLOR_AGENT_OVERALL_BEST = "purple"
COLOR_INFO_BG = "lightgrey"
COLOR_MAZE_OUTLINE = "#CCCCCC"
COLOR_BACKGROUND = "#282c34" 

COLOR_RANGEFINDER_RAY = "rgba(255, 255, 50, 0.3)" 
COLOR_RANGEFINDER_HIT = "rgba(255, 165, 50, 0.6)" 
COLOR_BEST_AGENT_TRAJECTORY = "cyan" 

class PlotWindow(tk.Toplevel):
    """Окреме вікно для відображення графіка."""
    def __init__(self, master, title="Plot"):
        super().__init__(master)
        self.title(title)
        self.figure = plt.Figure(figsize=(8, 6), dpi=100) 
        self.canvas = FigureCanvasTkAgg(self.figure, self) 
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_data(self, x_data, y_data, plot_title, x_label, y_label, line_label="Дані"):
        """Малює дані на графіку."""
        self.figure.clear() 
        ax = self.figure.add_subplot(111)
        ax.plot(x_data, y_data, label=line_label)
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if len(x_data) > 1 or len(y_data) > 1: 
            ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def on_close(self):
        plt.close(self.figure)
        self.destroy()

class MazeGUI:
    """Клас для графічного інтерфейсу симуляції NEAT у лабіринті."""

    def __init__(self, master: tk.Tk, config: dict, main_controller: 'SimulationController'):
        """
        Ініціалізація GUI.
        """
        self.master = master
        self.main_controller = main_controller
        self.config = config 
        self.master.title("NEAT Навігація в лабіринті")

        self._cell_size = config.get('CELL_SIZE_PX', 20)
        maze_width_px = config.get('MAZE_WIDTH', 21) * self._cell_size
        maze_height_px = config.get('MAZE_HEIGHT', 15) * self._cell_size
        info_panel_width = config.get('INFO_PANEL_WIDTH_PX', 300)


        self.is_running = False
        self.genome_to_display_id = None

        # --- Основні фрейми ---
        left_frame = tk.Frame(master)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.maze_frame = tk.Frame(left_frame, bd=1, relief=tk.SUNKEN)
        self.maze_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.control_frame = tk.Frame(master, width=info_panel_width, bg=COLOR_INFO_BG, bd=1, relief=tk.RAISED)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.pack_propagate(False)

        # --- Канвас для лабіринту ---
        self.maze_canvas = tk.Canvas(self.maze_frame, bg=COLOR_PATH,
                                     width=maze_width_px, height=maze_height_px,
                                     scrollregion=(0, 0, maze_width_px, maze_height_px))
        self.maze_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.best_agent_display_frame = tk.Frame(left_frame, bd=1, relief=tk.SUNKEN)
        self.best_agent_display_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(0,5)) 
        ttk.Label(self.best_agent_display_frame, text="Огляд найкращого агента", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=(2,2))
        best_agent_canvas_width = maze_width_px
        best_agent_canvas_height = maze_height_px
        self.best_agent_canvas = tk.Canvas(self.best_agent_display_frame, bg=COLOR_PATH,
                                           width=best_agent_canvas_width, height=best_agent_canvas_height)
        self.best_agent_canvas.pack(fill=tk.BOTH, expand=True)
        self.best_agent_canvas.addtag_all("best_agent_elements") 

        # Атрибути для візуалізації найкращого агента
        self.best_agent_trajectory_points = []
        self.best_agent_to_draw_instance: Optional[Agent] = None
        self.best_agent_display_maze: Optional[Maze] = None
        self.is_best_agent_viz_running = False
        self.current_best_agent_simulation_step = 0

        self.agent_tags = {}
        self.network_photo = None 
        self._current_network_pil_image: Optional[Image.Image] = None 
        # --- Атрибути для зуму та панорамування мережі ---
        self._network_zoom = 1.0
        self._network_offset_x = 0.0 
        self._network_offset_y = 0.0
        self._network_canvas_width = 1 
        self._network_canvas_height = 1 
        self._network_drag_start_x = 0
        self._network_drag_start_y = 0
        # --- Віджети на панелі керування ---
        self._create_control_widgets(self.control_frame)
        self.update_network_visualization()
        self._create_menubar() 

    def _create_menubar(self):
        """Створює головне меню програми (тулбар)."""
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        # --- Меню "File" ---
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        
        file_menu.add_command(label="Зберегти стан (JSON)", 
                            command=lambda: self.main_controller.save_simulation() if self.main_controller else None)
        file_menu.add_command(label="Загрузити стан (JSON)", 
                            command=lambda: self.main_controller.load_simulation() if self.main_controller else None)
        file_menu.add_separator()
        file_menu.add_command(label="Експортувати дані для CSV", command=self._export_current_data)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.master.quit)
        plots_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Графіки", menu=plots_menu)

        plots_menu.add_command(label="Середнє значення фітнесу за поколінням",
                            command=self._plot_avg_fitness)
        plots_menu.add_command(label="Максимальне значення фітнесу за поколінням",
                            command=self._plot_max_fitness)
        plots_menu.add_command(label="Кількість видів за поколінням",
                            command=self._plot_species_diversity)
        plots_menu.add_separator()
        plots_menu.add_command(label="Аналіз складності мереж",
                            command=self._plot_complexity_analysis)
        analysis_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Аналіз", menu=analysis_menu)
        
        analysis_menu.add_command(label="Аналіз поточного запуску (JSON)",
                                command=self._analyze_json_file)
        analysis_menu.add_command(label="Порівняти два запуски (JSON)",
                                command=self._compare_runs)

    def _plot_species_diversity(self):
        """Відображає графік кількості видів."""
        stats = self._get_plot_data()
        if not stats:
            messagebox.showinfo("Немає даних", "Немає даних про покоління для побудови графіка.")
            return

        generations = [s['generation'] for s in stats]
        num_species = [s.get('num_species', 0) for s in stats]

        plot_win = PlotWindow(self.master, title="Різноманіття видів за поколіннями")
        
        plot_win.figure.clear()
        ax = plot_win.figure.add_subplot(111)
        ax.plot(generations, num_species, label='Кількість видів', linewidth=2, color='green')
        ax.fill_between(generations, num_species, alpha=0.3, color='green')
        ax.set_xlabel('Покоління')
        ax.set_ylabel('Кількість видів')
        ax.set_title('Кількість видів за поколіннями')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_win.canvas.draw()

    def _plot_complexity_analysis(self):
        """Аналізує складність мереж у поточному запуску."""
        if not self.main_controller or not self.main_controller.neat:
            messagebox.showerror("Error", "Немає активної симуляції.")
            return
        
        complexity_data = []
        for genome in self.main_controller.neat.population:
            if genome:
                num_nodes = len(genome.nodes)
                num_connections = sum(1 for c in genome.connections.values() if c.enabled)
                complexity_data.append({
                    'nodes': num_nodes,
                    'connections': num_connections,
                    'fitness': genome.fitness
                })
        
        if not complexity_data:
            messagebox.showinfo("Немає даних", "Немає даних про складність мереж для поточного запуску.")
            return
        
        plot_win = PlotWindow(self.master, title="Аналіз складності мереж")
        plot_win.figure.clear()
        
        nodes = [d['nodes'] for d in complexity_data]
        connections = [d['connections'] for d in complexity_data]
        fitness = [d['fitness'] for d in complexity_data]
        
        ax = plot_win.figure.add_subplot(111)
        scatter = ax.scatter(nodes, connections, c=fitness, cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel('Кількість Вузлів')
        ax.set_ylabel('Кількість активних з\'єднань')
        ax.set_title('Складність мережі (за це покоління)')
        
        cbar = plot_win.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness')
        
        plot_win.canvas.draw()
    def _export_current_data(self):
        """Експортує поточні дані тренування в JSON."""
        if not self.main_controller or not self.main_controller.neat:
            messagebox.showerror("Error", "Немає активної симуляції.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Експортувати дані в JSON файл"
        )
        
        if filepath:
            try:
                from neat.json_serializer import NEATJSONSerializer
                NEATJSONSerializer.save_neat_state(
                    filepath, 
                    self.main_controller.neat, 
                    self.main_controller.config.copy()
                )
                messagebox.showinfo("Успіх", f"Дані експортовано в {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Не вдалось експортувати дані:\n{e}")

    def _analyze_json_file(self):
        """Завантажує та аналізує JSON файл."""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Виберіть JSON файл для аналізу"
        )
        
        if not filepath:
            return
        
        try:
            from neat.data_analyzer import NEATDataAnalyzer
            analyzer = NEATDataAnalyzer(filepath)
            
            analysis_window = tk.Toplevel(self.master)
            analysis_window.title(f"Аналіз: {os.path.basename(filepath)}")
            analysis_window.geometry("600x400")
            
            text_widget = tk.Text(analysis_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            info = analyzer.get_basic_info()
            text_widget.insert(tk.END, "=== Основна інформація ===\n")
            for key, value in info.items():
                text_widget.insert(tk.END, f"{key}: {value}\n")
            
            # Статистика фітнесу
            df = analyzer.get_fitness_statistics()
            if not df.empty:
                text_widget.insert(tk.END, "\n=== Статистика фітнесу ===\n")
                text_widget.insert(tk.END, f"Максимальне значення фітнесу: {df['max_fitness'].iloc[-1]:.4f}\n")
                text_widget.insert(tk.END, f"Середнє значення фітнесу: {df['max_fitness'].mean():.4f}\n")
                text_widget.insert(tk.END, f"Покращення фітнесу: {df['max_fitness'].iloc[-1] - df['max_fitness'].iloc[0]:.4f}\n")
            
            # Кнопки для графіків
            button_frame = ttk.Frame(analysis_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(button_frame, text="Показати графік фітнесу",
                    command=lambda: analyzer.plot_fitness_over_time()).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Показати графік видів", 
                    command=lambda: analyzer.plot_species_diversity()).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Показати графік складності мережі",
                    command=lambda: analyzer.plot_complexity_vs_fitness()).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалось проаналізувати файл:\n{e}")

    def _compare_runs(self):
        """Порівнює два запуски NEAT."""
        from tkinter import filedialog
        from neat.data_analyzer import NEATDataAnalyzer
        
        file1 = filedialog.askopenfilename(
            title="Виберіть Перший JSON Файл",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file1:
            return
        
        file2 = filedialog.askopenfilename(
            title="Виберіть Другий JSON Файл",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file2:
            return
        
        try:
            analyzer1 = NEATDataAnalyzer(file1)
            analyzer2 = NEATDataAnalyzer(file2)
            
            label1 = os.path.basename(file1).replace('.json', '')
            label2 = os.path.basename(file2).replace('.json', '')
            
            analyzer1.compare_runs(analyzer2, labels=(label1, label2))
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалось порівняти файли:\n{e}")
    def _get_plot_data(self) -> list:
        """Отримує дані для графіків з SimulationController."""
        if self.main_controller and hasattr(self.main_controller.neat, 'generation_statistics'):
            return self.main_controller.neat.generation_statistics
        return []

    def _plot_avg_fitness(self):
        """Відображає графік середнього фітнесу."""
        stats = self._get_plot_data()
        if not stats:
            messagebox.showinfo("Немає даних", "Немає даних про покоління для побудови графіка.")
            return

        generations = [s['generation'] for s in stats]
        avg_fitness = [s.get('average_fitness', 0) if s.get('average_fitness') is not None else 0 for s in stats] # Додав .get та обробку None

        plot_win = PlotWindow(self.master, title="Середній фітнес за поколіннями")
        plot_win.plot_data(generations, avg_fitness,
                            "Середній фітнес за поколіннями",
                            "Покоління", "Середній фітнес",
                            line_label="Середній фітнес")

    def _plot_max_fitness(self):
        """Відображає графік максимального фітнесу."""
        stats = self._get_plot_data()
        if not stats:
            messagebox.showinfo("Немає даних", "Немає даних про покоління для побудови графіка.")
            return

        generations = [s['generation'] for s in stats]
        max_fitness = [s.get('max_fitness', 0) if s.get('max_fitness') is not None else 0 for s in stats] # Додав .get та обробку None

        plot_win = PlotWindow(self.master, title="Максимальний фітнес за поколіннями")
        plot_win.plot_data(generations, max_fitness,
                            "Максимальний фітнес за поколіннями",
                            "Покоління", "Максимальний фітнес",
                            line_label="Максимальний фітнес")
        
    def _draw_rangefinder_rays(self, agent_id: int):
        """Малює промені rangefinder для конкретного агента."""
        agent = self.main_controller.agents.get(agent_id)
        if not agent or not hasattr(agent, 'last_rangefinder_rays'):
            return

        tag_rays = f"agent_{agent_id}_rays"
        self.maze_canvas.delete(tag_rays) 

        cell_s = self._cell_size
        
        ray_color_str = "yellow" 
        hit_point_radius = max(1, int(cell_s * 0.05))

        for sx, sy, ex, ey, actual_dist in agent.last_rangefinder_rays:
            sx_px, sy_px = sx * cell_s, sy * cell_s
            ex_px, ey_px = ex * cell_s, ey * cell_s
            
            line_width = 1 
            line_color = "yellow"
            if actual_dist < agent.rangefinder_max_dist - 0.1: # Якщо промінь щось зачепив
                line_color = "orange" 
            
            self.maze_canvas.create_line(sx_px, sy_px, ex_px, ey_px,
                                         fill=line_color, width=line_width, tags=(tag_rays, "rangefinder_ray"))
            
    def _update_species_inspector(self):
        """Оновлює Treeview інспектора видів поточними даними."""
        if not hasattr(self, 'species_tree') or not self.main_controller or not self.main_controller.neat:
            return

        expanded_species_iids = set()
        for species_iid in self.species_tree.get_children():
            if self.species_tree.item(species_iid, "open"):
                expanded_species_iids.add(species_iid)

        for item in self.species_tree.get_children():
            self.species_tree.delete(item)

        species_list = self.main_controller.neat.species
        if not species_list:
            self.species_tree.insert("", tk.END, text="Немає ще видів", iid="no_species_placeholder")
            return

        sorted_species_list = sorted(species_list, key=lambda s: s.id if s else float('inf'))

        for spec in sorted_species_list:
            if not spec:  
                continue

            rep_id_str = str(spec.representative.id) if spec.representative else "N/A"
            best_fit_str = f"{spec.best_fitness_ever:.4f}" if spec.best_fitness_ever is not None else "N/A"
            num_members = len(spec.members)
            
            species_iid = f"species_{spec.id}" # Унікальний ID для елемента Treeview

            self.species_tree.insert(
                "", tk.END,  # Вставляємо на верхній рівень
                iid=species_iid,
                text=str(spec.id),  
                values=(
                    num_members,
                    rep_id_str,
                    best_fit_str,
                    spec.generations_since_improvement
                ),
                open=(species_iid in expanded_species_iids) # Відновлюємо стан розгорнутості
            )

            if num_members > 0:
                sorted_members = sorted(spec.members, key=lambda g: g.fitness if g and g.fitness is not None else -float('inf'), reverse=True)
                max_genomes_to_show = 15 
                for i, member_genome in enumerate(sorted_members):
                    if i >= max_genomes_to_show and num_members > max_genomes_to_show:
                        self.species_tree.insert(species_iid, tk.END, 
                                                text=f"    ... and {num_members - max_genomes_to_show} more",
                                                values=("", "", "", ""))
                        break
                    if not member_genome: continue

                    genome_iid = f"genome_{spec.id}_{member_genome.id}"
                    fitness_str = f"{member_genome.fitness:.4f}" if member_genome.fitness is not None else "N/A"
                    self.species_tree.insert(
                        species_iid, tk.END, 
                        iid=genome_iid,
                        text=f"    └ GID: {member_genome.id}", 
                        values=(f"Fit: {fitness_str}", "-", "-", "-") 
                    )
            else:
                self.species_tree.insert(species_iid, tk.END, text="    (No members)")
    def _create_control_widgets(self, parent_frame):
        """Створює віджети на панелі керування."""
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent_frame, borderwidth=0, highlightthickness=0) 
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(5, 5)) 

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        scrollable_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _configure_scrollable_window(event):
            canvas.itemconfig(scrollable_window, width=event.width)

        canvas.bind("<Configure>", _configure_scrollable_window)

        canvas.grid(row=0, column=0, sticky="nswe") 
        scrollbar.grid(row=0, column=1, sticky="ns")

        
        scrollable_frame.grid_columnconfigure(0, weight=1)
        current_row = 0

         # --- Керування Симуляцією ---
        
        sim_control_frame = ttk.LabelFrame(scrollable_frame, text="Панель управління", padding=(5, 5))
        sim_control_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        sim_control_frame.columnconfigure(0, weight=1)
        self.start_pause_button = ttk.Button(sim_control_frame, text="Запустити візуалізацію", command=self._on_start_pause)
        self.start_pause_button.grid(row=0, column=0, sticky="ew", pady=2)
        self.next_gen_button = ttk.Button(sim_control_frame, text="Наступне покоління", command=self._on_next_generation, state=tk.NORMAL)
        self.next_gen_button.grid(row=1, column=0, sticky="ew", pady=2)
        
        run_n_frame = ttk.Frame(sim_control_frame)
        run_n_frame.grid(row=2, column=0, sticky="ew", pady=(5, 2))
        run_n_frame.columnconfigure(1, weight=1)

        ttk.Label(run_n_frame, text="Запуск").grid(row=0, column=0, padx=(0, 2))
        self.num_gens_var = tk.StringVar(value="50")
        self.num_gens_entry = ttk.Entry(run_n_frame, textvariable=self.num_gens_var, width=5)
        self.num_gens_entry.grid(row=0, column=1, padx=(0, 2), sticky="ew")
        ttk.Label(run_n_frame, text="Поколінь").grid(row=0, column=2, padx=(0, 5))
        self.run_n_button = ttk.Button(run_n_frame, text="ОК", width=4, command=self._on_run_n_generations)
        self.run_n_button.grid(row=0, column=3)

        self.reset_button = ttk.Button(sim_control_frame, text="Перезапустити симуляцію", command=self._on_reset)
        self.reset_button.grid(row=3, column=0, sticky="ew", pady=2)

        # --- Налаштування Лабіринту ---
        settings_frame = ttk.LabelFrame(scrollable_frame, text="Налаштування", padding=(5, 5))
        settings_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        settings_frame.columnconfigure(1, weight=1) 

        ttk.Label(settings_frame, text="Seed:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        initial_seed_value = self.config.get("MAZE_SEED", "")
        self.seed_var = tk.StringVar(value=str(initial_seed_value) if initial_seed_value is not None else "")
        self.seed_entry = ttk.Entry(settings_frame, textvariable=self.seed_var, width=15)
        self.seed_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.new_maze_button = ttk.Button(settings_frame, text="Згенерувати новий лабіринт", command=self._on_new_maze)
        self.new_maze_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 2))
        
        # --- Статистика ---
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Статистика", padding=(5, 5))
        stats_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        stats_frame.columnconfigure(0, weight=1)
        self.gen_label = ttk.Label(stats_frame, text="Поколінь: 0")
        self.gen_label.pack(anchor=tk.W)
        self.species_label = ttk.Label(stats_frame, text="Видів: 0")
        self.species_label.pack(anchor=tk.W)
        self.best_fitness_label = ttk.Label(stats_frame, text="Найкращий фітнес (покоління): N/A")
        self.best_fitness_label.pack(anchor=tk.W)
        self.avg_fitness_label = ttk.Label(stats_frame, text="Середній фітнес (покоління): N/A")
        self.avg_fitness_label.pack(anchor=tk.W)
        self.best_overall_fitness_label = ttk.Label(stats_frame, text="Найкращий фітнес (загалом): N/A")
        self.best_overall_fitness_label.pack(anchor=tk.W)
        self.goal_achieved_gen_label = ttk.Label(stats_frame, text="Досягнено цілі на поколінні: N/A") # <--- НОВИЙ LABEL
        self.goal_achieved_gen_label.pack(anchor=tk.W)

         # --- Save/Load Training ---
        save_load_frame = ttk.LabelFrame(scrollable_frame, text="Зберегти/Завантажити Тренування", padding=(5, 5))
        save_load_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        save_load_frame.columnconfigure(0, weight=1)
        save_load_frame.columnconfigure(1, weight=1)

        self.save_button = ttk.Button(save_load_frame, text="Зберегти стан", command=self._on_save_simulation)
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0,2), pady=2)
        self.load_button = ttk.Button(save_load_frame, text="Завантажити стан", command=self._on_load_simulation)
        self.load_button.grid(row=0, column=1, sticky="ew", padx=(2,0), pady=2)
        # --- Налаштування Візуалізації (сенсори) ---
        viz_settings_frame = ttk.LabelFrame(scrollable_frame, text="Параметри візуалізації", padding=(5,5))
        viz_settings_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        
        self.show_sensors_var = tk.BooleanVar(value=False)
        self.show_sensors_check = ttk.Checkbutton(viz_settings_frame, text="Показати сенсори агента",
                                                  variable=self.show_sensors_var,
                                                  command=self._on_toggle_show_sensors)
        self.show_sensors_check.pack(anchor=tk.W, padx=5, pady=2)
                # --- Інспектор Видів ---
        species_inspector_frame = ttk.LabelFrame(scrollable_frame, text="Іспектор видів", padding=(5, 5))
        species_inspector_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.grid_rowconfigure(current_row, weight=1) 
        current_row += 1
        species_inspector_frame.grid_rowconfigure(0, weight=1)
        species_inspector_frame.grid_columnconfigure(0, weight=1)

        # Створюємо Treeview
        self.species_tree = ttk.Treeview(species_inspector_frame, 
                                         columns=("s_members", "s_rep_id", "s_best_fit", "s_stagnant"), 
                                         show="tree headings") 

        self.species_tree.heading("#0", text="ID Видів") 
        self.species_tree.column("#0", width=70, minwidth=50, stretch=tk.NO, anchor=tk.W)

        self.species_tree.heading("s_members", text="Члени популяції")
        self.species_tree.column("s_members", width=60, minwidth=40, stretch=tk.NO, anchor=tk.CENTER)

        self.species_tree.heading("s_rep_id", text="Rep ID")
        self.species_tree.column("s_rep_id", width=60, minwidth=40, stretch=tk.NO, anchor=tk.CENTER)
        
        self.species_tree.heading("s_best_fit", text="Найкращ Фітнес за весь час")
        self.species_tree.column("s_best_fit", width=100, minwidth=70, stretch=tk.YES, anchor=tk.E)

        self.species_tree.heading("s_stagnant", text="Стагнація")
        self.species_tree.column("s_stagnant", width=60, minwidth=50, stretch=tk.NO, anchor=tk.CENTER)

        # Скролбари для Treeview
        species_tree_scrollbar_y = ttk.Scrollbar(species_inspector_frame, orient="vertical", command=self.species_tree.yview)
        species_tree_scrollbar_x = ttk.Scrollbar(species_inspector_frame, orient="horizontal", command=self.species_tree.xview)
        self.species_tree.configure(yscrollcommand=species_tree_scrollbar_y.set, xscrollcommand=species_tree_scrollbar_x.set)

        self.species_tree.grid(row=0, column=0, sticky="nsew")
        species_tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        species_tree_scrollbar_x.grid(row=1, column=0, sticky="ew")
        # --- Відображення Топології ---
        self.network_frame = ttk.LabelFrame(scrollable_frame, text="Топологія", padding=(5, 5))
        self.network_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.grid_rowconfigure(current_row, weight=1)
        current_row += 1

        self.network_canvas = tk.Canvas(self.network_frame, bg=COLOR_BACKGROUND, bd=0, highlightthickness=0)
        self.network_canvas.pack(fill=tk.BOTH, expand=True)
        self.network_canvas.bind("<Configure>", self._on_network_canvas_resize) 
        self.network_canvas.bind("<MouseWheel>", self._on_mouse_wheel)       
        self.network_canvas.bind("<Button-4>", self._on_mouse_wheel)        
        self.network_canvas.bind("<Button-5>", self._on_mouse_wheel)         
        self.network_canvas.bind("<ButtonPress-1>", self._on_network_drag_start)
        self.network_canvas.bind("<B1-Motion>", self._on_network_drag_motion)


        # --- Вибір Геному для Візуалізації ---
        select_frame = ttk.Frame(self.network_frame, padding=(0, 2))
        select_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(2,0))
        select_frame.columnconfigure(1, weight=1)

        ttk.Label(select_frame, text="ID:").grid(row=0, column=0, padx=(0, 2), sticky='w')
        self.genome_id_var = tk.StringVar()
        self.genome_id_entry = ttk.Entry(select_frame, textvariable=self.genome_id_var, width=8)
        self.genome_id_entry.grid(row=0, column=1, sticky='ew')
        self.visualize_id_button = ttk.Button(select_frame, text="ОК", width=5, command=self._on_visualize_genome_id)
        self.visualize_id_button.grid(row=0, column=2, padx=(2, 0))

    def _on_toggle_show_sensors(self):
        """Обробник зміни стану чекбокса "Show Agent Sensors"."""
        if not self.is_running and self.main_controller and hasattr(self.main_controller, '_update_agents_visuals'):
            self.main_controller._update_agents_visuals()
            self.update_gui() 
    def _on_save_simulation(self):
        if self.main_controller:
            if self.is_running: 
                self.is_running = False
                self.is_best_agent_viz_running = False 
                self.set_controls_state(False, running_multiple=self.main_controller._is_running_multiple if hasattr(self.main_controller, '_is_running_multiple') else False) # Оновлюємо стан кнопок
            self.main_controller.save_simulation()

    def _on_load_simulation(self):
        if self.main_controller:
            if self.is_running: 
                self.is_running = False
                self.is_best_agent_viz_running = False 
                self.set_controls_state(False)
            
            self.main_controller.load_simulation() 

            if hasattr(self.main_controller, 'config') and 'MAZE_SEED' in self.main_controller.config:
                loaded_seed = self.main_controller.config.get('MAZE_SEED')
                self.seed_var.set(str(loaded_seed) if loaded_seed is not None else "")
            else:
                 self.seed_var.set("")

            self.reset_best_agent_visualization()

            self.update_network_visualization()


    def _on_run_n_generations(self):
        if self.is_running: 
            messagebox.showwarning("Симуляція працює", "Не можна запускати кілька поколінь, поки симуляція працює. Зупиніть її спочатку.")
            return
        if self.is_best_agent_viz_running: 
            messagebox.showwarning("Найкраща симуляція працює", "Не можна запускати кілька поколінь, поки візуалізація найкращого агента працює. Зупиніть її спочатку.")
            return
        if not self.main_controller:
            print("Помилка: Контролер симуляції не ініціалізований.")
            return

        try:
            num_gens_to_run = int(self.num_gens_var.get())
            if num_gens_to_run <= 0:
                messagebox.showerror("Неправильне введення", "Кількість поколінь має бути додатнім числом.")
                return
        except ValueError:
            messagebox.showerror("Неправильне введення", f"Не дійсне число для покоління: '{self.num_gens_var.get()}'.")
            return

        self.set_controls_state(running=True, running_multiple=True)
        thread = threading.Thread(target=self.main_controller.run_multiple_generations,
                                 args=(num_gens_to_run,), daemon=True)
        thread.start()

    def _on_network_canvas_resize(self, event):
        """Обробник зміни розміру канвасу мережі."""
        self._network_canvas_width = event.width
        self._network_canvas_height = event.height
        self._redraw_network_image()
    def _on_mouse_wheel(self, event):
        """Обробник прокрутки колеса миші над канвасом мережі."""
        if not self._current_network_pil_image: return

        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor
        zoom_change = 0

        # Визначаємо напрямок прокрутки
        if event.num == 5 or event.delta < 0: # Zoom out
            zoom_change = zoom_out_factor
        elif event.num == 4 or event.delta > 0: # Zoom in 
            zoom_change = zoom_in_factor

        if zoom_change != 0:
            mouse_x, mouse_y = event.x, event.y

            img_w, img_h = self._current_network_pil_image.size
            if img_w <= 0 or img_h <= 0: return

            img_coord_x = (mouse_x - self._network_offset_x) / self._network_zoom
            img_coord_y = (mouse_y - self._network_offset_y) / self._network_zoom

            new_zoom = self._network_zoom * zoom_change
            new_zoom = max(0.1, min(new_zoom, 10.0))

            self._network_offset_x = mouse_x - img_coord_x * new_zoom
            self._network_offset_y = mouse_y - img_coord_y * new_zoom

            self._network_zoom = new_zoom
            self._redraw_network_image() 
    def _on_network_drag_start(self, event):
        """Початок перетягування зображення мережі."""
        self.network_canvas.scan_mark(event.x, event.y)
        self._network_drag_start_x = event.x
        self._network_drag_start_y = event.y

    def _on_network_drag_motion(self, event):
        """Переміщення під час перетягування зображення мережі."""
        if not self._current_network_pil_image: return

        dx = event.x - self._network_drag_start_x
        dy = event.y - self._network_drag_start_y

        self._network_offset_x += dx
        self._network_offset_y += dy

        self._network_drag_start_x = event.x
        self._network_drag_start_y = event.y

        self._redraw_network_image() 

    def draw_maze_on_best_agent_canvas(self, maze_grid: list[list[int]], cell_size: int):
        """Малює лабіринт на канвасі найкращого агента."""
        self.best_agent_canvas.delete("best_agent_maze") 

        height = len(maze_grid)
        width = len(maze_grid[0]) if height > 0 else 0
        if width == 0: return

        canvas_width = width * self._cell_size
        canvas_height = height * self._cell_size
        self.best_agent_canvas.config(scrollregion=(0, 0, canvas_width, canvas_height),
                                      width=canvas_width, height=canvas_height)

        for r in range(height):
            for c in range(width):
                x1, y1 = c * self._cell_size, r * self._cell_size
                x2, y2 = x1 + self._cell_size, y1 + self._cell_size
                fill_color = COLOR_PATH
                cell_type = maze_grid[r][c]

                if cell_type == CELL_WALL: fill_color = COLOR_WALL
                elif cell_type == CELL_START: fill_color = COLOR_START
                elif cell_type == CELL_GOAL: fill_color = COLOR_GOAL

                self.best_agent_canvas.create_rectangle(x1, y1, x2, y2,
                                                        fill=fill_color,
                                                        outline=COLOR_MAZE_OUTLINE,
                                                        tags=("best_agent_maze", "best_agent_elements"))

    def update_best_agent_visual_on_canvas(self, agent_instance: Agent, show_sensors: bool):
        """Оновлює візуальне представлення найкращого агента та його траєкторію."""
        if self._cell_size <= 0 or not agent_instance: return

        self.best_agent_canvas.delete("best_agent_visual") 

        radius = self._cell_size * 0.35
        x_pixel = agent_instance.x * self._cell_size
        y_pixel = agent_instance.y * self._cell_size
        color = COLOR_AGENT_OVERALL_BEST
        line_color = "white"

        x1, y1 = x_pixel - radius, y_pixel - radius
        x2, y2 = x_pixel + radius, y_pixel + radius
        line_len = radius
        end_x_heading = x_pixel + math.cos(agent_instance.angle) * line_len
        end_y_heading = y_pixel + math.sin(agent_instance.angle) * line_len

        self.best_agent_canvas.create_oval(x1, y1, x2, y2, fill=color, outline="black", width=1, tags=("best_agent_body", "best_agent_visual", "best_agent_elements"))
        self.best_agent_canvas.create_line(x_pixel, y_pixel, end_x_heading, end_y_heading, fill=line_color, width=max(1, int(self._cell_size * 0.1)), tags=("best_agent_heading", "best_agent_visual", "best_agent_elements"))

        current_pos_px = (x_pixel, y_pixel)
        if not self.best_agent_trajectory_points or \
           (abs(self.best_agent_trajectory_points[-1][0] - current_pos_px[0]) > 1e-3 or \
            abs(self.best_agent_trajectory_points[-1][1] - current_pos_px[1]) > 1e-3) :
            self.best_agent_trajectory_points.append(current_pos_px)

        max_traj_points = 200
        if len(self.best_agent_trajectory_points) > max_traj_points:
            self.best_agent_trajectory_points = self.best_agent_trajectory_points[-max_traj_points:]

        if len(self.best_agent_trajectory_points) > 1:
            self.best_agent_canvas.create_line(
                self.best_agent_trajectory_points,
                fill=COLOR_BEST_AGENT_TRAJECTORY,
                width=2,
                tags=("best_agent_trajectory_line", "best_agent_visual", "best_agent_elements")
            )
        
        # Сенсори для найкращого агента
        if show_sensors and hasattr(agent_instance, 'last_rangefinder_rays'):
            cell_s = self._cell_size
            for sx, sy, ex, ey, actual_dist in agent_instance.last_rangefinder_rays:
                sx_px, sy_px = sx * cell_s, sy * cell_s
                ex_px, ey_px = ex * cell_s, ey * cell_s
                ray_line_color = "yellow"
                if actual_dist < agent_instance.rangefinder_max_dist - 0.1 : ray_line_color = "orange"
                self.best_agent_canvas.create_line(sx_px, sy_px, ex_px, ey_px,
                                                   fill=ray_line_color, width=1, 
                                                   tags=("best_agent_sensors", "best_agent_visual", "best_agent_elements"))
        
        self.best_agent_canvas.tag_raise("best_agent_body") 
        self.best_agent_canvas.tag_raise("best_agent_heading")


    def clear_best_agent_trajectory(self):
        """Очищає траєкторію найкращого агента."""
        self.best_agent_trajectory_points = []
        self.best_agent_canvas.delete("best_agent_trajectory_line")

    def reset_best_agent_visualization(self):
        """Повністю скидає візуалізацію найкращого агента."""
        print("Resetting best agent visualization.")
        self.is_best_agent_viz_running = False 
        self.best_agent_canvas.delete("best_agent_elements") 
        self.best_agent_trajectory_points = []
        self.best_agent_to_draw_instance = None
        self.best_agent_display_maze = None
        self.current_best_agent_simulation_step = 0

    def update_network_visualization(self, genome_id_to_show: Optional[int] = None):
         """Оновлює візуалізацію мережі для заданого ID (або найкращого)."""
         genome_to_visualize = None
         actual_genome_id = None

         if genome_id_to_show is not None and self.main_controller:
             genome_to_visualize = self.main_controller.get_genome_by_id(genome_id_to_show)
             if genome_to_visualize: actual_genome_id = genome_id_to_show
         if not genome_to_visualize and self.main_controller:
             best_overall = self.main_controller.neat.best_genome_overall
             if best_overall:
                  genome_to_visualize = best_overall
                  actual_genome_id = best_overall.id
             elif self.main_controller.neat.population:
                  valid_pop = [g for g in self.main_controller.neat.population if g is not None]
                  if valid_pop:
                       best_current = max(valid_pop, key=lambda g: g.fitness, default=None)
                       if best_current:
                            genome_to_visualize = best_current
                            actual_genome_id = best_current.id
             if not genome_to_visualize and self.main_controller.neat.population:
                   valid_pop = [g for g in self.main_controller.neat.population if g is not None]
                   if valid_pop:
                       genome_to_visualize = random.choice(valid_pop)
                       actual_genome_id = genome_to_visualize.id

         title_text = "Топологія мережі"
         if actual_genome_id is not None:
             title_text += f" (ID Геному: {actual_genome_id})"
             self.genome_id_var.set(str(actual_genome_id))
         else:
             self.genome_id_var.set("")
         self.genome_to_display_id = actual_genome_id 

         if isinstance(self.network_frame, ttk.LabelFrame):
              self.network_frame.config(text=title_text)

         network_image_pil = None
         if genome_to_visualize:
               network_image_pil = visualize_network(genome_to_visualize, zoom_factor=1.0) 

         self.display_network(network_image_pil) 
    def draw_maze(self, maze_grid: list[list[int]], cell_size: int):
        """Малює лабіринт на канвасі."""
        self._cell_size = cell_size
        self.maze_canvas.delete("maze") 

        height = len(maze_grid)
        width = len(maze_grid[0]) if height > 0 else 0
        if width == 0: return

        canvas_width = width * cell_size
        canvas_height = height * cell_size
        self.maze_canvas.config(scrollregion=(0, 0, canvas_width, canvas_height),
                               width=canvas_width, height=canvas_height) 

        for r in range(height):
            for c in range(width):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                fill_color = COLOR_PATH
                cell_type = maze_grid[r][c]

                if cell_type == CELL_WALL: fill_color = COLOR_WALL
                elif cell_type == CELL_START: fill_color = COLOR_START
                elif cell_type == CELL_GOAL: fill_color = COLOR_GOAL

                self.maze_canvas.create_rectangle(x1, y1, x2, y2,
                                                  fill=fill_color,
                                                  outline=COLOR_MAZE_OUTLINE,
                                                  tags="maze")

    def update_agent_visual(self, agent_id: int, x: float, y: float, angle_rad: float,
                              is_best_gen: bool = False, is_best_overall: bool = False,
                              show_sensors: bool = False): 
        """Оновлює або створює візуальне представлення агента ТА його сенсорів."""
        if self._cell_size <= 0: return

        radius = self._cell_size * 0.35
        x_pixel = x * self._cell_size
        y_pixel = y * self._cell_size

        if is_best_overall: color, line_color, z = COLOR_AGENT_OVERALL_BEST, "white", 3
        elif is_best_gen: color, line_color, z = COLOR_AGENT_BEST, "black", 2
        else: color, line_color, z = COLOR_AGENT_DEFAULT, "white", 1

        x1, y1 = x_pixel - radius, y_pixel - radius
        x2, y2 = x_pixel + radius, y_pixel + radius
        line_len = radius
        end_x_heading = x_pixel + math.cos(angle_rad) * line_len
        end_y_heading = y_pixel + math.sin(angle_rad) * line_len

        tag_agent_body = f"agent_{agent_id}_body"
        tag_agent_heading = f"agent_{agent_id}_heading"
        
        self.maze_canvas.delete(tag_agent_body)
        self.maze_canvas.delete(tag_agent_heading)

        self.maze_canvas.create_oval(x1, y1, x2, y2, fill=color, outline="black", width=1, tags=(tag_agent_body, "agent_body"))
        self.maze_canvas.create_line(x_pixel, y_pixel, end_x_heading, end_y_heading, fill=line_color, width=max(1, int(self._cell_size * 0.1)), tags=(tag_agent_heading, "agent_heading"))

        for _ in range(z):
             self.maze_canvas.tag_raise(tag_agent_body)
             self.maze_canvas.tag_raise(tag_agent_heading)
        
        tag_rays = f"agent_{agent_id}_rays"
        self.maze_canvas.delete(tag_rays) 

        if show_sensors:
            agent = self.main_controller.agents.get(agent_id)
            if agent and hasattr(agent, 'last_rangefinder_rays'):
                cell_s = self._cell_size
                for sx, sy, ex, ey, actual_dist in agent.last_rangefinder_rays:
                    sx_px, sy_px = sx * cell_s, sy * cell_s
                    ex_px, ey_px = ex * cell_s, ey * cell_s
                    
                    ray_line_color = "yellow"
                    ray_line_width = 1
                    if actual_dist < agent.rangefinder_max_dist - 0.1:
                        ray_line_color = "orange"
                    
                    self.maze_canvas.create_line(sx_px, sy_px, ex_px, ey_px,
                                                 fill=ray_line_color, width=ray_line_width, tags=(tag_rays, "rangefinder_ray"))
                    self.maze_canvas.tag_lower(tag_rays, tag_agent_body)


    def remove_agent_visual(self, agent_id: int):
        """Видаляє візуальне представлення агента."""
        tag = self.agent_tags.pop(agent_id, None)
        if tag:
            self.maze_canvas.delete(tag)

    def clear_all_agents(self):
        """Видаляє всіх агентів (тіла, напрямки, промені) з канвасу."""
        self.maze_canvas.delete("agent_body")
        self.maze_canvas.delete("agent_heading")
        self.maze_canvas.delete("rangefinder_ray") 


    def update_stats(self, generation: int, num_species: int, best_fitness_gen: Optional[float], avg_fitness_gen: Optional[float], best_fitness_overall: Optional[float], best_genome_id_to_show: Optional[int]):
        """Оновлює текстові мітки зі статистикою та ініціює візуалізацію мережі."""
        self.gen_label.config(text=f"Покоління: {generation}")
        self.species_label.config(text=f"Видів: {num_species}")
        bfg_text = f"{best_fitness_gen:.4f}" if best_fitness_gen is not None else "N/A"
        afg_text = f"{avg_fitness_gen:.4f}" if avg_fitness_gen is not None else "N/A"
        bfo_text = f"{best_fitness_overall:.4f}" if best_fitness_overall is not None else "N/A"
        self.best_fitness_label.config(text=f"Найкращий фітнес (Покоління): {bfg_text}")
        self.avg_fitness_label.config(text=f"Середній фітнес (Покоління): {afg_text}")
        self.best_overall_fitness_label.config(text=f"Найкращий фітнес (Загалом): {bfo_text}")

        # Оновлення візуалізації мережі
        genome_to_visualize = None
        if best_genome_id_to_show is not None and self.main_controller:
            genome_to_visualize = self.main_controller.get_genome_by_id(best_genome_id_to_show)

        title_text = "Топологія мережі"
        if genome_to_visualize:
             title_text += f" (ID Геному: {genome_to_visualize.id})"
             self.genome_id_var.set(str(genome_to_visualize.id))
        else:
             self.genome_id_var.set("")

        if isinstance(self.network_frame, ttk.LabelFrame):
             self.network_frame.config(text=title_text)

       
        self.update_network_visualization(self.genome_to_display_id)

    def display_network(self, network_image_pil: Optional[Image.Image]):
        """Зберігає ОРИГІНАЛЬНЕ PIL зображення мережі та ініціює перемалювання."""
        if network_image_pil is not None and self._current_network_pil_image is not network_image_pil:
             self._network_zoom = 1.0
             if hasattr(self, 'network_canvas'): 
                 self._network_offset_x = self._network_canvas_width / 2 - (network_image_pil.width / 2) if network_image_pil else 0
                 self._network_offset_y = self._network_canvas_height / 2 - (network_image_pil.height / 2) if network_image_pil else 0
             else:
                 self._network_offset_x = 0
                 self._network_offset_y = 0

        self._current_network_pil_image = network_image_pil
        self._redraw_network_image()
    def _redraw_network_image(self, event=None):
        """
        Перемальовує зображення мережі на канвасі з урахуванням
        поточного масштабу (_network_zoom) та зсуву (_network_offset_x/y).
        """
        self.network_canvas.delete("network_img")
        if not self._current_network_pil_image:
            canvas_width = self._network_canvas_width
            canvas_height = self._network_canvas_height
            if canvas_width > 1 and canvas_height > 1:
                 self.network_canvas.create_text(canvas_width / 2, canvas_height / 2, text="Не існує мережі", fill="grey", anchor=tk.CENTER, tags="network_img")
            return

        if self._network_canvas_width <= 1: self._network_canvas_width = self.network_canvas.winfo_width()
        if self._network_canvas_height <= 1: self._network_canvas_height = self.network_canvas.winfo_height()
        canvas_width = self._network_canvas_width
        canvas_height = self._network_canvas_height
        if canvas_width <= 1 or canvas_height <= 1: return 

        img_original = self._current_network_pil_image
        original_width, original_height = img_original.size
        if original_width <= 0 or original_height <= 0: return

        # Розрахунок розміру масштабованого зображення 
        display_width = int(original_width * self._network_zoom)
        display_height = int(original_height * self._network_zoom)

        if display_width <= 0 or display_height <= 0: return 

        try:
             img_resized = img_original.resize((display_width, display_height), Image.Resampling.LANCZOS)
             self.network_photo = ImageTk.PhotoImage(img_resized)
        except Exception as e:
             print(f"Помилка зміни розміру фото: {e}")
             try: self.network_photo = ImageTk.PhotoImage(img_original) 
             except: return

        draw_x = self._network_offset_x
        draw_y = self._network_offset_y

        self.network_canvas.create_image(draw_x, draw_y, anchor=tk.NW, 
                                         image=self.network_photo, tags="network_img")


    def update_gui_from_thread(self, stats):
         """Безпечно оновлює GUI зі стану, переданого з іншого потоку."""
         self.master.after(0, self._update_gui_safe, stats)
    def _update_gui_safe(self, stats):
         """Метод, що фактично оновлює GUI."""
         if not stats: return
         try:
              self.gen_label.config(text=f"Покоління: {stats.get('generation', 'N/A')}")
              self.species_label.config(text=f"Видів: {stats.get('num_species', 'N/A')}")
              bfg = stats.get('max_fitness')
              afg = stats.get('average_fitness')
              bfo = stats.get('best_overall_fitness')
              bfg_text = f"{bfg:.4f}" if bfg is not None else "N/A"
              afg_text = f"{afg:.4f}" if afg is not None else "N/A"
              bfo_text = f"{bfo:.4f}" if bfo is not None else "N/A"
              self.best_fitness_label.config(text=f"Найкращий фітнес (Покоління): {bfg_text}")
              self.avg_fitness_label.config(text=f"Середній фітнес (Покоління): {afg_text}")
              self.best_overall_fitness_label.config(text=f"Найкращий фітнес (Загалом): {bfo_text}")

              first_goal_gen = stats.get('first_goal_achieved_generation') 
              fgg_text = str(first_goal_gen) if first_goal_gen is not None else "N/A"
              self.goal_achieved_gen_label.config(text=f"Ціль досягнено на поколінні: {fgg_text}") 

              # Оновлення візуалізації мережі
              id_to_show_on_network_canvas = self.genome_to_display_id # ID, обраний користувачем
              if id_to_show_on_network_canvas is None: 
                if stats.get('best_genome_overall') and stats['best_genome_overall'].id is not None:
                    id_to_show_on_network_canvas = stats['best_genome_overall'].id
                elif stats.get('best_genome_current_gen') and stats['best_genome_current_gen'].id is not None:
                    id_to_show_on_network_canvas = stats['best_genome_current_gen'].id

              self.update_network_visualization(id_to_show_on_network_canvas)

              self._update_species_inspector() 

              self.update_gui() 
         except Exception as e:
              print(f"Error updating GUI from thread: {e}")

    def set_controls_state(self, running: bool, running_multiple: bool = False):
         """Встановлює стан кнопок керування."""
         self.is_running = running # Стан візуалізації
         # Блокуємо все, якщо виконується run_multiple_generations
         sim_controls_state = tk.DISABLED if running_multiple else tk.NORMAL
         viz_controls_state = tk.DISABLED if running_multiple or running else tk.NORMAL

         self.start_pause_button.config(text="Призупинити" if running else "Запустити візуалізацію", state=sim_controls_state if not running_multiple else tk.DISABLED)
         self.next_gen_button.config(state=sim_controls_state)
         self.run_n_button.config(state=sim_controls_state)
         self.num_gens_entry.config(state=sim_controls_state)
         self.reset_button.config(state=sim_controls_state)
         self.new_maze_button.config(state=sim_controls_state)
         self.visualize_id_button.config(state=viz_controls_state)
         self.genome_id_entry.config(state=viz_controls_state)

    def _on_start_pause(self):
        if self.main_controller:
            self.is_running = not self.is_running 
            
            if self.is_running:
                if self.main_controller.neat.generation > 0 and self.main_controller.neat.best_genome_overall:
                    print("Starting best agent visualization.")
                    self.is_best_agent_viz_running = True
                    self.current_best_agent_simulation_step = 0 
                    self.best_agent_canvas.delete("best_agent_visual") 
                    self.best_agent_canvas.delete("best_agent_sensors")
                    self.clear_best_agent_trajectory()
                    self.master.after(30, self.best_agent_simulation_step) 
                else:
                    print("Best agent visualization not started (gen 0 or no best genome).")
                    self.is_best_agent_viz_running = False
                    self.reset_best_agent_visualization() 
            else: 
                print("Pausing best agent visualization.")
                self.is_best_agent_viz_running = False
            
            self.set_controls_state(self.is_running, 
                                    running_multiple=getattr(self.main_controller, '_is_running_multiple', False))
            self.main_controller.toggle_simulation(self.is_running) 
        else: print("Помилка: Відсутній головний контролер (main_controller).")

    def _on_next_generation(self):
        if self.is_running or self.is_best_agent_viz_running: 
            messagebox.showwarning("Симуляцію активна", "Призупиніть поточну візуалізацію перед переходом до наступного покоління.")
            return

        if not self.main_controller:
            print("Помилка: Відсутній головний контролер (main_controller).")
            return
        
        is_batch_running = getattr(self.main_controller, '_is_running_multiple', False)
        self.set_controls_state(True, running_multiple=is_batch_running) 
        self.master.update()
        try:
            self.main_controller.run_one_generation()
            self.update_network_visualization() 
            self.reset_best_agent_visualization()
        finally:
            self.set_controls_state(False, running_multiple=is_batch_running)

    def _on_new_maze(self):
         if self.is_running or self.is_best_agent_viz_running: 
             messagebox.showwarning("Симуляцію активна", "Призупиніть поточну візуалізацію перед створенням нового лабіринту.")
             return
         if not self.main_controller:
             print("Помилка: Відсутній головний контролер (main_controller).")
             return
            
         seed_str = self.seed_var.get().strip()
         seed = None
         if seed_str:
             try: seed = int(seed_str)
             except ValueError:
                 messagebox.showerror("Invalid Seed", f"Cannot parse seed: '{seed_str}'. Using random.")
                 self.seed_var.set("")
         
         new_seed_used = self.main_controller.generate_new_maze(seed)
         
         if new_seed_used is not None:
             self.seed_var.set(str(new_seed_used))
         
         self.reset_best_agent_visualization() 
    def _on_reset(self):
        if self.is_running or self.is_best_agent_viz_running: 
            messagebox.showwarning("Симуляція активна", "Призупиніть поточну візуалізацію перед скиданням.")
            return
        if not self.main_controller:
            print("Помилка: Відсутній головний контролер (main_controller).")
            return

        if messagebox.askyesno("Підтвердіть", "Ви впевнені, що хочете скинути симуляцію?"):
            is_batch_running = getattr(self.main_controller, '_is_running_multiple', False)
            self.set_controls_state(True, running_multiple=is_batch_running) 
            self.master.update()
            try:
                self.main_controller.reset_simulation()
                new_seed_from_controller = self.main_controller.config.get('MAZE_SEED', '')
                self.seed_var.set(str(new_seed_from_controller) if new_seed_from_controller is not None else "")
                self.reset_best_agent_visualization() 
                self.update_network_visualization()
            finally:
                self.set_controls_state(False, running_multiple=is_batch_running)

    def _on_visualize_genome_id(self):
        """Обробник кнопки 'Show' для візуалізації геному за ID."""
        if self.is_running or self.is_best_agent_viz_running: return 
        if not self.main_controller: return

        genome_id_str = self.genome_id_var.get().strip()
        print(genome_id_str)
        if not genome_id_str:
            messagebox.showwarning("Помилка введення", "Будь ласка, введіть ID геному для візуалізації.")
            return

        try:
            genome_id_to_find = int(genome_id_str) 
            self.update_network_visualization(genome_id_to_find) 
            if self.genome_to_display_id is None or int(self.genome_to_display_id) != genome_id_to_find:
                messagebox.showerror("Геном не знайдено", f"Геному з ID'{genome_id_to_find}' не знайдено в поточній популяції.")
        except ValueError:
            messagebox.showerror("Помилковий ID", f"'{genome_id_str}'. Повинен бути цілим числом.")
        except Exception as e:
            messagebox.showerror("Помилка візуалізації", f"Не вдається візуалізувати {genome_id_str}:\n{e}")
            import traceback
            traceback.print_exc()
    def best_agent_simulation_step(self):
        if not self.is_best_agent_viz_running or not self.main_controller:
            self.is_best_agent_viz_running = False 
            return

        best_genome = self.main_controller.neat.best_genome_overall
        current_maze_seed_from_controller = self.main_controller.config.get('MAZE_SEED')

        if not best_genome or self.main_controller.neat.generation == 0:
            self.is_best_agent_viz_running = False
            self.reset_best_agent_visualization()
            return

        if (self.best_agent_to_draw_instance is None) or \
           (self.best_agent_to_draw_instance.id != best_genome.id) or \
           (self.best_agent_display_maze and self.best_agent_display_maze.seed != current_maze_seed_from_controller):
            
            print(f"Best Agent Viz: Initializing/Updating for genome {best_genome.id} on maze seed {current_maze_seed_from_controller}")
            
            self.best_agent_display_maze = Maze(
                self.main_controller.config['MAZE_WIDTH'],
                self.main_controller.config['MAZE_HEIGHT'],
                current_maze_seed_from_controller
            )
            self.draw_maze_on_best_agent_canvas(self.best_agent_display_maze.grid, self._cell_size)
            
            self.best_agent_to_draw_instance = Agent(
                best_genome.id, 
                self.best_agent_display_maze.start_pos,
                self.main_controller.config.copy()
            )
            self.clear_best_agent_trajectory() 
            self.current_best_agent_simulation_step = 0 

        agent = self.best_agent_to_draw_instance
        maze = self.best_agent_display_maze
        
        if agent.reached_goal or self.current_best_agent_simulation_step >= self.main_controller.config.get('MAX_STEPS_PER_EVALUATION', 400):
            self.update_best_agent_visual_on_canvas(agent, self.show_sensors_var.get())
            if self.is_best_agent_viz_running: 
                 self.master.after(30, self.best_agent_simulation_step)
            return

        if not agent.reached_goal:
            sensor_readings = agent.get_sensor_readings(maze)
            try:
                network_outputs = activate_network(best_genome.copy(), sensor_readings) 
                if network_outputs is None:
                    raise ValueError(f"activate_network returned None for best agent genome {best_genome.id}")
                
                agent.update(maze, network_outputs, dt=1.0) 
                agent.steps_taken = self.current_best_agent_simulation_step + 1 
            except Exception as e:
                print(f"Error in best_agent_simulation_step (update) for genome {best_genome.id}: {e}")
                import traceback
                traceback.print_exc()
                self.is_best_agent_viz_running = False 
                return
        
        self.update_best_agent_visual_on_canvas(agent, self.show_sensors_var.get())
        self.current_best_agent_simulation_step += 1
        
        if self.is_best_agent_viz_running:
             self.master.after(30, self.best_agent_simulation_step) # Швидкість візуалізації (мс)
    def update_gui(self):
         """Оновлює GUI Tkinter."""
         try:
            self.master.update()
            self.master.update_idletasks()
         except tk.TclError as e:
              if "application has been destroyed" not in str(e):
                   print(f"Tkinter update error: {e}")