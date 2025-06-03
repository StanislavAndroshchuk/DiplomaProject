import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd


class NEATDataAnalyzer:
    """Клас для аналізу та візуалізації даних NEAT з JSON файлів."""
    
    def __init__(self, json_filepath: str):
        """Завантажує дані з JSON файлу."""
        with open(json_filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get("metadata", {})
        self.generation_history = self.data.get("generation_history", [])
        self.all_genomes = self.data.get("all_genomes", {})
        self.config = self.data.get("config", {})
    
    def get_basic_info(self) -> Dict:
        """Повертає базову інформацію про збережені дані."""
        return {
            "save_date": self.metadata.get("save_date"),
            "generation_saved": self.metadata.get("generation_saved"),
            "total_genomes": self.metadata.get("total_genomes"),
            "total_generations": self.metadata.get("total_generations"),
            "maze_seed": self.metadata.get("maze_seed"),
            "population_size": self.config.get("POPULATION_SIZE")
        }
    
    def get_fitness_statistics(self) -> pd.DataFrame:
        """Повертає статистику фітнесу для всіх поколінь як DataFrame."""
        stats_data = []
        
        for gen_data in self.generation_history:
            gen_num = gen_data.get("generation", 0)
            max_fit = gen_data.get("max_fitness")
            avg_fit = gen_data.get("average_fitness")
            num_species = gen_data.get("num_species", 0)
            best_genome_id = gen_data.get("best_genome_current_gen_id")
            
            stats_data.append({
                "generation": gen_num,
                "max_fitness": max_fit,
                "average_fitness": avg_fit,
                "num_species": num_species,
                "best_genome_id": best_genome_id
            })
        
        return pd.DataFrame(stats_data)
    
    def plot_fitness_over_time(self, save_path: Optional[str] = None):
        """Малює графік максимального та середнього фітнесу."""
        df = self.get_fitness_statistics()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['generation'], df['max_fitness'], label='Max Fitness', linewidth=2)
        plt.plot(df['generation'], df['average_fitness'], label='Average Fitness', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_species_diversity(self, save_path: Optional[str] = None):
        """Малює графік кількості видів."""
        df = self.get_fitness_statistics()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['generation'], df['num_species'], label='Number of Species', linewidth=2, color='green')
        plt.fill_between(df['generation'], df['num_species'], alpha=0.3, color='green')
        
        plt.xlabel('Generation')
        plt.ylabel('Number of Species')
        plt.title('Species Diversity Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_genome_complexity_evolution(self) -> pd.DataFrame:
        """Аналізує еволюцію складності геномів."""
        complexity_data = []
        
        for genome_id, genome_data in self.all_genomes.items():
            num_nodes = len(genome_data.get("nodes", {}))
            num_connections = len(genome_data.get("connections", {}))
            enabled_connections = sum(1 for conn in genome_data.get("connections", {}).values() 
                                    if conn.get("enabled", False))
            
            complexity_data.append({
                "genome_id": int(genome_id),
                "num_nodes": num_nodes,
                "num_connections": num_connections,
                "enabled_connections": enabled_connections,
                "fitness": genome_data.get("fitness", 0)
            })
        
        return pd.DataFrame(complexity_data)
    
    def plot_complexity_vs_fitness(self, save_path: Optional[str] = None):
        """Малює залежність між складністю мережі та фітнесом."""
        df = self.get_genome_complexity_evolution()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Кількість вузлів vs фітнес
        ax1.scatter(df['num_nodes'], df['fitness'], alpha=0.5)
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Network Size vs Fitness')
        ax1.grid(True, alpha=0.3)
        
        # Кількість з'єднань vs фітнес
        ax2.scatter(df['enabled_connections'], df['fitness'], alpha=0.5, color='orange')
        ax2.set_xlabel('Number of Active Connections')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Connection Count vs Fitness')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_genomes_per_generation(self) -> List[Dict]:
        """Повертає найкращі геноми для кожного покоління."""
        best_genomes = []
        
        for gen_data in self.generation_history:
            gen_num = gen_data.get("generation", 0)
            max_fitness = gen_data.get("max_fitness")
            best_genome_id = gen_data.get("best_genome_current_gen_id")
            
            if best_genome_id and str(best_genome_id) in self.all_genomes:
                genome_data = self.all_genomes[str(best_genome_id)]
                best_genomes.append({
                    "generation": gen_num,
                    "genome_id": best_genome_id,
                    "fitness": max_fitness,
                    "num_nodes": len(genome_data.get("nodes", {})),
                    "num_connections": len(genome_data.get("connections", {}))
                })
        
        return best_genomes
    
    def compare_runs(self, other_analyzer: 'NEATDataAnalyzer', labels: Tuple[str, str] = ("Run 1", "Run 2")):
        """Порівнює два запуски NEAT."""
        df1 = self.get_fitness_statistics()
        df2 = other_analyzer.get_fitness_statistics()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Максимальний фітнес
        ax1.plot(df1['generation'], df1['max_fitness'], label=f'{labels[0]} - Max', linewidth=2)
        ax1.plot(df2['generation'], df2['max_fitness'], label=f'{labels[1]} - Max', linewidth=2, linestyle='--')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Max Fitness')
        ax1.set_title('Maximum Fitness Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Кількість видів
        ax2.plot(df1['generation'], df1['num_species'], label=f'{labels[0]} - Species', linewidth=2)
        ax2.plot(df2['generation'], df2['num_species'], label=f'{labels[1]} - Species', linewidth=2, linestyle='--')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Species')
        ax2.set_title('Species Diversity Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, output_dir: str):
        """Експортує дані у CSV файли для подальшого аналізу."""
        import os
        
        # Створюємо директорію якщо не існує
        os.makedirs(output_dir, exist_ok=True)
        
        # Експортуємо статистику фітнесу
        fitness_df = self.get_fitness_statistics()
        fitness_df.to_csv(os.path.join(output_dir, "fitness_statistics.csv"), index=False)
        
        # Експортуємо дані про складність
        complexity_df = self.get_genome_complexity_evolution()
        complexity_df.to_csv(os.path.join(output_dir, "genome_complexity.csv"), index=False)
        
        # Експортуємо найкращі геноми
        best_genomes_df = pd.DataFrame(self.get_best_genomes_per_generation())
        best_genomes_df.to_csv(os.path.join(output_dir, "best_genomes.csv"), index=False)
        
        print(f"Data exported to {output_dir}")
