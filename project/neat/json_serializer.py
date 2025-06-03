import json
from datetime import datetime
from typing import List

from neat.genome import ConnectionGene, Genome, NodeGene

class NEATJSONEncoder(json.JSONEncoder):
    """Спеціальний JSON encoder для NEAT об'єктів."""
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


class NEATJSONSerializer:
    """Клас для серіалізації та десеріалізації NEAT даних у JSON формат."""
    
    VERSION = "1.0" # Нова версія якщо буде Novelty Search
    
    @staticmethod
    def serialize_node_gene(node_gene) -> dict:
        """Серіалізує NodeGene в словник."""
        return {
            "id": node_gene.id,
            "type": node_gene.type,
            "bias": node_gene.bias,
            "activation_function": node_gene.activation_function_name
        }
    
    @staticmethod
    def deserialize_node_gene(data: dict, NodeGene) -> 'NodeGene':
        """Десеріалізує NodeGene зі словника."""
        return NodeGene(
            node_id=data["id"],
            node_type=data["type"],
            bias=data.get("bias"),
            activation_func=data.get("activation_function", "sigmoid")
        )
    
    @staticmethod
    def serialize_connection_gene(conn_gene) -> dict:
        """Серіалізує ConnectionGene в словник."""
        return {
            "in_node_id": conn_gene.in_node_id,
            "out_node_id": conn_gene.out_node_id,
            "weight": conn_gene.weight,
            "enabled": conn_gene.enabled,
            "innovation": conn_gene.innovation
        }
    
    @staticmethod
    def deserialize_connection_gene(data: dict, ConnectionGene) -> 'ConnectionGene':
        """Десеріалізує ConnectionGene зі словника."""
        return ConnectionGene(
            in_node_id=data["in_node_id"],
            out_node_id=data["out_node_id"],
            weight=data["weight"],
            enabled=data["enabled"],
            innovation_num=data["innovation"]
        )
    
    @staticmethod
    def serialize_genome(genome) -> dict:
        """Серіалізує Genome в словник."""
        return {
            "id": genome.id,
            "fitness": genome.fitness,
            "adjusted_fitness": genome.adjusted_fitness,
            "species_id": genome.species_id,
            "nodes": {
                str(node_id): NEATJSONSerializer.serialize_node_gene(node)
                for node_id, node in genome.nodes.items()
            },
            "connections": {
                str(innov): NEATJSONSerializer.serialize_connection_gene(conn)
                for innov, conn in genome.connections.items()
            },
            "input_node_ids": genome._input_node_ids,
            "output_node_ids": genome._output_node_ids,
            "bias_node_id": genome._bias_node_id
        }
    
    @staticmethod
    def deserialize_genome(data: dict, config: dict, Genome, NodeGene, ConnectionGene, innovation_manager) -> 'Genome':
        """Десеріалізує Genome зі словника."""
        # Створюємо порожній геном
        genome = Genome(data["id"], 0, 0, config, innovation_manager)
        genome.nodes.clear()
        genome.connections.clear()
        
        # Відновлюємо атрибути
        genome.fitness = data["fitness"]
        genome.adjusted_fitness = data["adjusted_fitness"]
        genome.species_id = data.get("species_id")
        genome._input_node_ids = data["input_node_ids"]
        genome._output_node_ids = data["output_node_ids"]
        genome._bias_node_id = data.get("bias_node_id")
        
        # Відновлюємо вузли
        for node_id_str, node_data in data["nodes"].items():
            node = NEATJSONSerializer.deserialize_node_gene(node_data, NodeGene)
            genome.nodes[int(node_id_str)] = node
        
        # Відновлюємо з'єднання
        for innov_str, conn_data in data["connections"].items():
            conn = NEATJSONSerializer.deserialize_connection_gene(conn_data, ConnectionGene)
            genome.connections[int(innov_str)] = conn
        
        return genome
    
    @staticmethod
    def serialize_species(species) -> dict:
        """Серіалізує Species в словник."""
        return {
            "id": species.id,
            "representative_id": species.representative.id if species.representative else None,
            "member_ids": [member.id for member in species.members if member],
            "generations_since_improvement": species.generations_since_improvement,
            "best_fitness_ever": species.best_fitness_ever,
            "offspring_count": species.offspring_count
        }
    
    @staticmethod
    def serialize_generation_data(generation: int, population: List, species: List, stats: dict) -> dict:
        """Серіалізує дані одного покоління."""
        # Серіалізуємо тільки ID геномів, а не самі об'єкти
        serialized_stats = {
            "generation": stats.get("generation", generation),
            "max_fitness": stats.get("max_fitness"),
            "average_fitness": stats.get("average_fitness"),
            "num_species": stats.get("num_species", len(species)),
            "num_species_after_speciation": stats.get("num_species_after_speciation"),
            "best_genome_current_gen_id": None, 
            "best_genome_overall_id": None,     
            "first_goal_achieved_generation": stats.get("first_goal_achieved_generation")
        }
        
        if "best_genome_current_gen" in stats and stats["best_genome_current_gen"]:
            genome = stats["best_genome_current_gen"]
            if hasattr(genome, 'id'):
                serialized_stats["best_genome_current_gen_id"] = genome.id
            else:
                serialized_stats["best_genome_current_gen_id"] = None
        
        if "best_genome_overall" in stats and stats["best_genome_overall"]:
            genome = stats["best_genome_overall"]
            if hasattr(genome, 'id'):
                serialized_stats["best_genome_overall_id"] = genome.id
            else:
                serialized_stats["best_genome_overall_id"] = None
        
        return serialized_stats
    
    @staticmethod
    def save_neat_state(filepath: str, neat_algorithm, config: dict):
        """Зберігає повний стан NEAT алгоритму в JSON файл."""
        all_genomes = {}
        
        # Додаємо геноми з поточної популяції
        for genome in neat_algorithm.population:
            if genome:
                all_genomes[genome.id] = genome
        
        for species in neat_algorithm.species:
            if species.representative:
                all_genomes[species.representative.id] = species.representative
            for member in species.members:
                if member:
                    all_genomes[member.id] = member
        
        if neat_algorithm.best_genome_overall:
            all_genomes[neat_algorithm.best_genome_overall.id] = neat_algorithm.best_genome_overall
        
        # з попередніх представників
        for rep in neat_algorithm.species_representatives_prev_gen.values():
            if rep:
                all_genomes[rep.id] = rep
        
        # Структура даних
        data = {
            "version": NEATJSONSerializer.VERSION,
            "metadata": {
                "save_date": datetime.now().isoformat(),
                "generation_saved": neat_algorithm.generation,
                "maze_seed": config.get("MAZE_SEED"),
                "total_genomes": len(all_genomes),
                "total_generations": len(neat_algorithm.generation_statistics)
            },
            "config": config,
            "innovation_manager": {
                "next_node_id": neat_algorithm.innovation_manager._next_node_id,
                "next_innovation_num": neat_algorithm.innovation_manager._next_innovation_num
            },
            "all_genomes": {
                str(gid): NEATJSONSerializer.serialize_genome(genome)
                for gid, genome in all_genomes.items()
            },
            "generation_history": [
                NEATJSONSerializer.serialize_generation_data(
                    stat_entry.get("generation", 0),
                    [],  
                    [],  
                    stat_entry
                ) for stat_entry in neat_algorithm.generation_statistics
            ],
            "current_state": {
                "generation": neat_algorithm.generation,
                "population_ids": [g.id for g in neat_algorithm.population if g],
                "species": [NEATJSONSerializer.serialize_species(s) for s in neat_algorithm.species if s],
                "best_genome_overall_id": neat_algorithm.best_genome_overall.id if neat_algorithm.best_genome_overall else None,
                "species_representatives_prev_gen": {
                    str(sid): rep.id for sid, rep in neat_algorithm.species_representatives_prev_gen.items()
                },
                "genome_id_counter": next(neat_algorithm._genome_id_counter) - 1,
                "max_species_id": max((s.id for s in neat_algorithm.species), default=0) if neat_algorithm.species else 0,
                "first_goal_achieved_generation": neat_algorithm.first_goal_achieved_generation 
            }
        }
        
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NEATJSONEncoder)
    
    @staticmethod
    def load_neat_state(filepath: str, config: dict, NeatAlgorithm, Genome, NodeGene, ConnectionGene, Species, InnovationManager):
        """Завантажує стан NEAT алгоритму з JSON файлу."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get("version") != NEATJSONSerializer.VERSION:
            print(f"Warning: JSON version mismatch. File: {data.get('version')}, Expected: {NEATJSONSerializer.VERSION}")
        
        # Оновлюємо конфіг
        saved_config = data.get("config", {})
        config.update(saved_config)
        
        # Створюємо NEAT алгоритм
        num_inputs = config["NUM_INPUTS"]
        num_outputs = config["NUM_OUTPUTS"]
        initial_genome_id = data["current_state"]["genome_id_counter"] + 1
        neat = NeatAlgorithm(config, num_inputs, num_outputs, initial_genome_id_start=initial_genome_id, _is_loading=True)
        
        # Відновлюємо innovation manager
        im_data = data["innovation_manager"]
        neat.innovation_manager._next_node_id = im_data["next_node_id"]
        neat.innovation_manager._next_innovation_num = im_data["next_innovation_num"]
        
        # Відновлюємо всі геноми
        all_genomes = {}
        for gid_str, genome_data in data["all_genomes"].items():
            genome = NEATJSONSerializer.deserialize_genome(
                genome_data, config, Genome, NodeGene, ConnectionGene, neat.innovation_manager
            )
            all_genomes[int(gid_str)] = genome
        
        # Відновлюємо поточний стан
        current_state = data["current_state"]
        neat.generation = current_state["generation"]
        neat.first_goal_achieved_generation = current_state.get("first_goal_achieved_generation") # <--- ЗАВАНТАЖУЄМО
        
        # Відновлюємо популяцію
        neat.population = [
            all_genomes[gid] for gid in current_state["population_ids"]
            if gid in all_genomes
        ]
        
        # Відновлюємо найкращий геном
        best_id = current_state.get("best_genome_overall_id")
        neat.best_genome_overall = all_genomes.get(best_id) if best_id else None
        
        # Відновлюємо види
        max_species_id = current_state["max_species_id"]
        Species._species_counter = itertools.count(max_species_id + 1)
        
        neat.species = []
        for species_data in current_state["species"]:
            rep_id = species_data["representative_id"]
            if rep_id and rep_id in all_genomes:
                species = Species(all_genomes[rep_id])
                species.id = species_data["id"]
                species.generations_since_improvement = species_data["generations_since_improvement"]
                species.best_fitness_ever = species_data["best_fitness_ever"]
                species.offspring_count = species_data["offspring_count"]
                
                # Додаємо членів
                species.members = []
                for member_id in species_data["member_ids"]:
                    if member_id in all_genomes:
                        species.add_member(all_genomes[member_id])
                
                neat.species.append(species)
        
        # Відновлюємо попередніх представників
        neat.species_representatives_prev_gen = {}
        for sid_str, rep_id in current_state["species_representatives_prev_gen"].items():
            if rep_id in all_genomes:
                neat.species_representatives_prev_gen[int(sid_str)] = all_genomes[rep_id]
        
        # Відновлюємо історію поколінь
        # Конвертуємо серіалізовані дані назад у формат, який очікує NEAT
        generation_history = []
        for hist_entry_data in data.get("generation_history", []): 
            stat_entry = {
                "generation": hist_entry_data.get("generation", 0),
                "max_fitness": hist_entry_data.get("max_fitness"),
                "average_fitness": hist_entry_data.get("average_fitness"),
                "num_species": hist_entry_data.get("num_species", 0),
                "num_species_after_speciation": hist_entry_data.get("num_species_after_speciation"),
                "best_genome_current_gen": None,
                "best_genome_overall": None,
                "first_goal_achieved_generation": hist_entry_data.get("first_goal_achieved_generation") 
            }
            
            best_current_id = hist_entry_data.get("best_genome_current_gen_id")
            if best_current_id and best_current_id in all_genomes:
                stat_entry["best_genome_current_gen"] = all_genomes[best_current_id]
            
            best_overall_id = hist_entry_data.get("best_genome_overall_id")
            if best_overall_id and best_overall_id in all_genomes:
                stat_entry["best_genome_overall"] = all_genomes[best_overall_id]
            
            generation_history.append(stat_entry)
        
        neat.generation_statistics = generation_history
        
        return neat


import itertools