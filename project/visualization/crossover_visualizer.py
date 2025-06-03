import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import random
from PIL import ImageTk

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from neat.genome import Genome, NodeGene, ConnectionGene
from neat.innovation import InnovationManager
from visualization.network_visualizer import visualize_network


class CrossoverVisualizer:
    """Візуалізатор процесу кросоверу NEAT."""
    
    def __init__(self, master):
        self.master = master
        self.master.title("NEAT Crossover Visualizer")
        self.master.geometry("1400x900")
        
        self.config = {
            'INHERIT_DISABLED_GENE_RATE': 0.75,
            'WEIGHT_INIT_RANGE': 1.0,
            'C1_EXCESS': 1.0,
            'C2_DISJOINT': 1.0,
            'C3_WEIGHT': 0.4,
        }
        
        self.innovation_manager = InnovationManager(start_node_id=10, start_innovation_num=0)
        self.parent1 = None
        self.parent2 = None
        self.child = None
        
        self._create_ui()
        self._create_example_genomes()
        
    def _create_ui(self):
        """Створює інтерфейс користувача."""
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Generate Random Parents", 
                   command=self._create_random_parents).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Create Example Parents", 
                   command=self._create_example_genomes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Create Disjoint Example", 
                   command=self._create_disjoint_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Perform Crossover", 
                   command=self._perform_crossover).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Parent 1 Fitness:").pack(side=tk.LEFT, padx=(20, 5))
        self.fitness1_var = tk.DoubleVar(value=100.0)
        ttk.Entry(control_frame, textvariable=self.fitness1_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Parent 2 Fitness:").pack(side=tk.LEFT, padx=(10, 5))
        self.fitness2_var = tk.DoubleVar(value=80.0)
        ttk.Entry(control_frame, textvariable=self.fitness2_var, width=10).pack(side=tk.LEFT, padx=5)
        
        parent1_frame = ttk.LabelFrame(main_frame, text="Parent 1", padding="10")
        parent1_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        parent2_frame = ttk.LabelFrame(main_frame, text="Parent 2", padding="10")
        parent2_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        child_frame = ttk.LabelFrame(main_frame, text="Offspring", padding="10")
        child_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.canvas1 = tk.Canvas(parent1_frame, width=300, height=200, bg="white")
        self.canvas1.pack()
        
        self.canvas2 = tk.Canvas(parent2_frame, width=300, height=200, bg="white")
        self.canvas2.pack()
        
        self.canvas_child = tk.Canvas(child_frame, width=300, height=200, bg="white")
        self.canvas_child.pack()
        
        self.text1 = tk.Text(parent1_frame, width=40, height=15, font=("Consolas", 9))
        self.text1.pack(pady=5)
        
        self.text2 = tk.Text(parent2_frame, width=40, height=15, font=("Consolas", 9))
        self.text2.pack(pady=5)
        
        self.text_child = tk.Text(child_frame, width=40, height=15, font=("Consolas", 9))
        self.text_child.pack(pady=5)
        
        alignment_frame = ttk.LabelFrame(main_frame, text="Gene Alignment", padding="10")
        alignment_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.alignment_canvas = tk.Canvas(alignment_frame, width=1300, height=200, bg="white")
        self.alignment_canvas.pack()
        
        # Налаштування ваги рядків/колонок
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure((0, 1, 2), weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def _create_example_genomes(self):
        """Створює приклад геномів з документації NEAT."""
        # Parent 1: складніша структура
        self.parent1 = Genome(1, 3, 1, self.config, self.innovation_manager)
        self.parent1.fitness = self.fitness1_var.get()
        self.parent1.connections.clear()
        hidden_node = NodeGene(5, "HIDDEN", bias=0.0)
        self.parent1.add_node(hidden_node)
        self.parent1.add_connection(ConnectionGene(1, 4, 0.7, True, 1))
        self.parent1.add_connection(ConnectionGene(2, 4, -0.5, False, 2))  # Disabled
        self.parent1.add_connection(ConnectionGene(3, 4, 0.5, True, 3))
        self.parent1.add_connection(ConnectionGene(2, 5, 0.2, True, 4))
        self.parent1.add_connection(ConnectionGene(5, 4, 0.4, True, 5))
        self.parent1.add_connection(ConnectionGene(1, 5, 0.6, True, 8))
        
        # Parent 2: інша структура
        self.parent2 = Genome(2, 3, 1, self.config, self.innovation_manager)
        self.parent2.fitness = self.fitness2_var.get()
        self.parent2.connections.clear()
        self.parent2.add_node(NodeGene(5, "HIDDEN", bias=0.0))
        self.parent2.add_node(NodeGene(6, "HIDDEN", bias=0.0))
        self.parent2.add_connection(ConnectionGene(1, 4, 0.8, True, 1))
        self.parent2.add_connection(ConnectionGene(2, 4, -0.2, True, 2))
        self.parent2.add_connection(ConnectionGene(3, 4, 0.9, True, 3))
        self.parent2.add_connection(ConnectionGene(2, 5, 0.3, True, 4))
        self.parent2.add_connection(ConnectionGene(5, 4, 0.5, True, 5))
        self.parent2.add_connection(ConnectionGene(5, 6, 0.1, True, 6))
        self.parent2.add_connection(ConnectionGene(6, 4, 0.2, True, 7))
        self.parent2.add_connection(ConnectionGene(3, 5, 0.7, True, 9))
        self.parent2.add_connection(ConnectionGene(1, 6, 0.4, True, 10))
        
        self._update_display()
        
    def _create_disjoint_example(self):
        """Створює приклад з явними disjoint генами."""
        self.innovation_manager = InnovationManager(start_node_id=10, start_innovation_num=0)
        
        # Parent 1
        self.parent1 = Genome(1, 3, 1, self.config, self.innovation_manager)
        self.parent1.fitness = self.fitness1_var.get()
        self.parent1.connections.clear()
        self.parent1.add_connection(ConnectionGene(0, 4, 0.5, True, 1))
        self.parent1.add_connection(ConnectionGene(1, 4, 0.3, True, 2))
        self.parent1.add_connection(ConnectionGene(2, 4, -0.2, True, 3))
        self.parent1.add_connection(ConnectionGene(3, 4, 0.8, True, 5))  
        hidden1 = NodeGene(5, "HIDDEN", bias=0.0)
        self.parent1.add_node(hidden1)
        self.parent1.add_connection(ConnectionGene(1, 5, 0.4, True, 7))  
        self.parent1.add_connection(ConnectionGene(5, 4, 0.6, True, 8))
        
        # Parent 2
        self.parent2 = Genome(2, 3, 1, self.config, self.innovation_manager)
        self.parent2.fitness = self.fitness2_var.get()
        self.parent2.connections.clear()
        self.parent2.add_connection(ConnectionGene(0, 4, 0.6, True, 1))
        self.parent2.add_connection(ConnectionGene(1, 4, 0.4, True, 2))
        self.parent2.add_connection(ConnectionGene(2, 4, -0.3, True, 3))
        hidden2 = NodeGene(6, "HIDDEN", bias=0.0)
        self.parent2.add_node(hidden2)
        self.parent2.add_connection(ConnectionGene(0, 6, 0.7, True, 4))  
        self.parent2.add_connection(ConnectionGene(2, 6, 0.5, True, 6))  
        self.parent2.add_connection(ConnectionGene(6, 4, 0.3, True, 9))
        self.parent2.add_connection(ConnectionGene(3, 6, 0.2, True, 10))
        
        self._update_display()
        
    def _create_random_parents(self):
        """Створює випадкових батьків з можливістю disjoint генів."""
        manager1 = InnovationManager(start_node_id=10, start_innovation_num=0)
        manager2 = InnovationManager(start_node_id=10, start_innovation_num=0)
        
        # Parent 1
        self.parent1 = Genome(1, 3, 1, self.config, manager1)
        self.parent1.fitness = self.fitness1_var.get()
        for _ in range(random.randint(1, 3)):
            self.parent1.mutate_add_connection(manager1)
        for _ in range(random.randint(1, 2)):
            if random.random() < 0.5:
                self.parent1.mutate_add_node(manager1)
            else:
                self.parent1.mutate_add_connection(manager1)
        
        # Parent 2
        self.parent2 = Genome(2, 3, 1, self.config, manager2)
        self.parent2.fitness = self.fitness2_var.get()
        for _ in range(random.randint(1, 3)):
            self.parent2.mutate_add_connection(manager2)
        for _ in range(random.randint(1, 2)):
            if random.random() < 0.5:
                self.parent2.mutate_add_node(manager2)
            else:
                self.parent2.mutate_add_connection(manager2)
                
        self._synchronize_innovations()
        
        # Використовуємо загальний менеджер для подальшого кросоверу
        self.innovation_manager = InnovationManager(
            start_node_id=max(manager1._next_node_id, manager2._next_node_id),
            start_innovation_num=max(manager1._next_innovation_num, manager2._next_innovation_num)
        )
                
        self._update_display()
        
    def _synchronize_innovations(self):
        """Синхронізує інноваційні номери для спільних структур між батьками."""
        # Знаходимо з'єднання з однаковою структурою
        for conn1 in self.parent1.connections.values():
            for conn2 in self.parent2.connections.values():
                if (conn1.in_node_id == conn2.in_node_id and 
                    conn1.out_node_id == conn2.out_node_id):
                    min_innov = min(conn1.innovation, conn2.innovation)
                    
                    # Оновлюємо в обох батьків
                    if conn1.innovation != min_innov:
                        old_innov = conn1.innovation
                        new_conn = ConnectionGene(conn1.in_node_id, conn1.out_node_id, 
                                                conn1.weight, conn1.enabled, min_innov)
                        del self.parent1.connections[old_innov]
                        self.parent1.connections[min_innov] = new_conn
                        
                    if conn2.innovation != min_innov:
                        old_innov = conn2.innovation
                        new_conn = ConnectionGene(conn2.in_node_id, conn2.out_node_id,
                                                conn2.weight, conn2.enabled, min_innov)
                        del self.parent2.connections[old_innov]
                        self.parent2.connections[min_innov] = new_conn
        
    def _perform_crossover(self):
        """Виконує кросовер між батьками."""
        if not self.parent1 or not self.parent2:
            messagebox.showwarning("Warning", "Please create parent genomes first!")
            return
            
        self.parent1.fitness = self.fitness1_var.get()
        self.parent2.fitness = self.fitness2_var.get()
        g1_is_fitter = self.parent1.fitness >= self.parent2.fitness
        
        self.child = self._corrected_crossover(self.parent1, self.parent2, g1_is_fitter)
        
        self._update_display()
        self._visualize_alignment()
        
    def _corrected_crossover(self, genome1: Genome, genome2: Genome, g1_is_fitter: bool) -> Genome:
        """Виправлена версія кросоверу згідно з правилами NEAT."""
        config = genome1.config
        
        # Створюємо нащадка
        child = Genome(f"c_{genome1.id}_{genome2.id}", 
                      len(genome1._input_node_ids), 
                      len(genome1._output_node_ids), 
                      config, 
                      self.innovation_manager)
        
        # Спочатку копіюємо ВСІ базові вузли
        for node_id in genome1._input_node_ids:
            if node_id in genome1.nodes:
                child.nodes[node_id] = genome1.nodes[node_id].copy()
        for node_id in genome1._output_node_ids:
            if node_id in genome1.nodes:
                child.nodes[node_id] = genome1.nodes[node_id].copy()
        if genome1._bias_node_id is not None and genome1._bias_node_id in genome1.nodes:
            child.nodes[genome1._bias_node_id] = genome1.nodes[genome1._bias_node_id].copy()
        
        # правильні списки ID
        child._input_node_ids = list(genome1._input_node_ids)
        child._output_node_ids = list(genome1._output_node_ids)
        child._bias_node_id = genome1._bias_node_id
        
        child.connections.clear()
        
        innovs1 = set(genome1.connections.keys())
        innovs2 = set(genome2.connections.keys())
        
        max_innov1 = max(innovs1) if innovs1 else 0
        max_innov2 = max(innovs2) if innovs2 else 0
        
        all_innovs = innovs1.union(innovs2)
        
        # Класифікуємо гени
        matching = innovs1.intersection(innovs2)
        disjoint1 = {i for i in innovs1 if i not in innovs2 and i <= max_innov2}
        disjoint2 = {i for i in innovs2 if i not in innovs1 and i <= max_innov1}
        excess1 = {i for i in innovs1 if i not in innovs2 and i > max_innov2}
        excess2 = {i for i in innovs2 if i not in innovs1 and i > max_innov1}
        
        for innov in sorted(all_innovs):
            if innov in matching:
                # вибираємо випадково
                conn1 = genome1.connections[innov]
                conn2 = genome2.connections[innov]
                chosen_conn = random.choice([conn1, conn2]).copy()
                
                if not conn1.enabled or not conn2.enabled:
                    disable_prob = config.get('INHERIT_DISABLED_GENE_RATE', 0.75)
                    chosen_conn.enabled = (random.random() >= disable_prob)
                    
                child.add_connection(chosen_conn)
                
            elif g1_is_fitter and innov in (disjoint1.union(excess1)):
                # Disjoint/excess від кращого батька
                child.add_connection(genome1.connections[innov].copy())
                
            elif not g1_is_fitter and innov in (disjoint2.union(excess2)):
                # Disjoint/excess від кращого батька
                child.add_connection(genome2.connections[innov].copy())
                
            elif genome1.fitness == genome2.fitness:
                # При однаковому фітнесі - випадково
                if innov in innovs1:
                    child.add_connection(genome1.connections[innov].copy())
                elif innov in innovs2:
                    child.add_connection(genome2.connections[innov].copy())
        
        required_hidden_nodes = set()
        for conn in child.connections.values():
            if conn.in_node_id not in child.nodes:
                required_hidden_nodes.add(conn.in_node_id)
            if conn.out_node_id not in child.nodes:
                required_hidden_nodes.add(conn.out_node_id)
                
        for node_id in required_hidden_nodes:
            if node_id in genome1.nodes:
                child.add_node(genome1.nodes[node_id].copy())
            elif node_id in genome2.nodes:
                child.add_node(genome2.nodes[node_id].copy())
                
        return child
        
    def _visualize_alignment(self):
        """Візуалізує вирівнювання генів."""
        if not self.parent1 or not self.parent2 or not self.child:
            return
            
        self.alignment_canvas.delete("all")
        
        # Параметри візуалізації
        gene_width = 80
        gene_height = 40
        spacing = 10
        y_parent1 = 20
        y_parent2 = 80
        y_child = 140
        
        all_innovs = sorted(set(self.parent1.connections.keys()) | 
                          set(self.parent2.connections.keys()))
        
        if not all_innovs:
            return
            
        max_innov1 = max(self.parent1.connections.keys()) if self.parent1.connections else 0
        max_innov2 = max(self.parent2.connections.keys()) if self.parent2.connections else 0
        
        # Малюємо мітки
        self.alignment_canvas.create_text(5, y_parent1 + gene_height/2, 
                                        text="P1", anchor="w", font=("Arial", 10, "bold"))
        self.alignment_canvas.create_text(5, y_parent2 + gene_height/2, 
                                        text="P2", anchor="w", font=("Arial", 10, "bold"))
        self.alignment_canvas.create_text(5, y_child + gene_height/2, 
                                        text="Child", anchor="w", font=("Arial", 10, "bold"))
        
        # Малюємо гени
        for i, innov in enumerate(all_innovs):
            x = 40 + i * (gene_width + spacing)
            
            # Parent 1
            if innov in self.parent1.connections:
                conn = self.parent1.connections[innov]
                color = self._get_gene_color(innov, 1, max_innov1, max_innov2)
                self._draw_gene(x, y_parent1, gene_width, gene_height, 
                              conn, color, "P1")
                
            # Parent 2
            if innov in self.parent2.connections:
                conn = self.parent2.connections[innov]
                color = self._get_gene_color(innov, 2, max_innov1, max_innov2)
                self._draw_gene(x, y_parent2, gene_width, gene_height, 
                              conn, color, "P2")
                
            # Child
            if innov in self.child.connections:
                conn = self.child.connections[innov]
                from_parent = "?"
                if innov in self.parent1.connections and innov in self.parent2.connections:
                    if abs(conn.weight - self.parent1.connections[innov].weight) < 0.001:
                        from_parent = "P1"
                    else:
                        from_parent = "P2"
                elif innov in self.parent1.connections:
                    from_parent = "P1"
                elif innov in self.parent2.connections:
                    from_parent = "P2"
                    
                color = "#90EE90"  
                self._draw_gene(x, y_child, gene_width, gene_height, 
                              conn, color, from_parent)
                
    def _get_gene_color(self, innov, parent_num, max1, max2):
        """Визначає колір гена залежно від типу."""
        in1 = innov in self.parent1.connections
        in2 = innov in self.parent2.connections
        
        if in1 and in2:
            return "#87CEEB"  # Matching - light blue
        elif parent_num == 1:
            if innov > max2:
                return "#FFB6C1"  # Excess - light pink
            else:
                return "#FFA07A"  # Disjoint - light orange
        else:
            if innov > max1:
                return "#FFB6C1"  # Excess - light pink
            else:
                return "#FFA07A"  # Disjoint - light orange
            
    def _draw_gene(self, x, y, width, height, conn, color, label):
        """Малює ген з'єднання."""
        self.alignment_canvas.create_rectangle(
            x, y, x + width, y + height,
            fill=color, outline="black", width=2
        )
        
        text = f"In {conn.innovation}\n{conn.in_node_id}->{conn.out_node_id}\nW:{conn.weight:.2f}"
        if not conn.enabled:
            text += "\nDISAB"
            
        self.alignment_canvas.create_text(
            x + width/2, y + height/2,
            text=text, font=("Consolas", 8)
        )
        
        self.alignment_canvas.create_text(
            x + width/2, y - 5,
            text=label, font=("Consolas", 8, "bold")
        )
        
    def _update_display(self):
        """Оновлює відображення всіх геномів."""
        self._display_phenotype(self.parent1, self.canvas1)
        self._display_phenotype(self.parent2, self.canvas2)
        self._display_phenotype(self.child, self.canvas_child)
        
        self._display_genotype(self.parent1, self.text1)
        self._display_genotype(self.parent2, self.text2)
        self._display_genotype(self.child, self.text_child)
        
    def _display_phenotype(self, genome, canvas):
        """Відображає фенотип (мережу) на canvas."""
        canvas.delete("all")
        
        if not genome:
            return
            
        try:
            network_image = visualize_network(genome, zoom_factor=0.8)
            if network_image:
                photo = ImageTk.PhotoImage(network_image)
                canvas.create_image(150, 100, image=photo)
                canvas.image = photo  
        except Exception as e:
            print(f"Error visualizing network: {e}")
            canvas.create_text(150, 100, text="Visualization Error", 
                             font=("Arial", 12), fill="red")
            
    def _display_genotype(self, genome, text_widget):
        """Відображає генотип у текстовому віджеті."""
        text_widget.delete(1.0, tk.END)
        
        if not genome:
            return
            
        text_widget.insert(tk.END, f"Genome ID: {genome.id}\n", "header")
        text_widget.insert(tk.END, f"Fitness: {genome.fitness}\n\n", "header")
        text_widget.insert(tk.END, "NODES:\n", "section")
        for node_id in sorted(genome.nodes.keys()):
            node = genome.nodes[node_id]
            text_widget.insert(tk.END, f"  {node}\n")
        text_widget.insert(tk.END, "\nCONNECTIONS:\n", "section")
        for innov in sorted(genome.connections.keys()):
            conn = genome.connections[innov]
            status = "E" if conn.enabled else "D"
            text_widget.insert(tk.END, 
                f"  [{innov}] {conn.in_node_id}->{conn.out_node_id} "
                f"w={conn.weight:.3f} [{status}]\n")
            
        text_widget.tag_config("header", font=("Consolas", 10, "bold"))
        text_widget.tag_config("section", font=("Consolas", 9, "bold"), 
                              foreground="blue")


def main():
    """Запускає візуалізатор кросоверу."""
    root = tk.Tk()
    app = CrossoverVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()