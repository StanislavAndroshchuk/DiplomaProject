import random
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed 
import os
from typing import Optional 

from .genome import Genome
from .innovation import InnovationManager
from .species import Species

class NeatAlgorithm:
    """
    Клас, що реалізує основний цикл алгоритму NEAT:
    Оцінка -> Видоутворення -> Відбір -> Розмноження (Кросовер + Мутація).
    """

    def __init__(self, config: dict, num_inputs: int, num_outputs: int, initial_genome_id_start=0, _is_loading=False):
        self.config = config
        self._genome_id_counter = itertools.count(initial_genome_id_start)
        self.population_size = config.get('POPULATION_SIZE', 150)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.species_representatives_prev_gen: dict[int, Genome] = {} 
        self.generation_statistics: list[dict] = [] # Для збереження історії
        required_keys = [
            'POPULATION_SIZE', 'COMPATIBILITY_THRESHOLD', 'C1_EXCESS',
            'C2_DISJOINT', 'C3_WEIGHT', 'WEIGHT_MUTATE_RATE',
            'WEIGHT_REPLACE_RATE', 'WEIGHT_MUTATE_POWER', 'WEIGHT_CAP',
            'WEIGHT_INIT_RANGE', 'ADD_CONNECTION_RATE', 'ADD_NODE_RATE',
            'ELITISM', 'SELECTION_PERCENTAGE', 'CROSSOVER_RATE',
            'MAX_STAGNATION', 'INHERIT_DISABLED_GENE_RATE'
        ]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
             raise ValueError(f"Missing required configuration key(s) in config: {', '.join(missing_keys)}")
        node_id_for_innov_manager = num_inputs + num_outputs + 1 
        self.innovation_manager = InnovationManager(
            start_node_id=node_id_for_innov_manager,
            start_innovation_num=0 # Починаємо з 0, Genome.__init__ використає менеджер
        )

        self._genome_id_counter = itertools.count(0) # Глобальний лічильник ID геномів
        
        self.population = self._create_initial_population(self.innovation_manager)
        self.species = []
        self.generation = 0
        self.best_genome_overall = None
        self.first_goal_achieved_generation: Optional[int] = None 

        self._speciate_population()
        if not _is_loading:
            self.population = self._create_initial_population(self.innovation_manager)
            self._speciate_population() # Перше видоутворення використовує випадкових представників з поточної популяції
            self._update_previous_gen_representatives() # Зберігаємо представників для наступного покоління
    
    def _update_previous_gen_representatives(self):
        """Оновлює словник представників попереднього покоління."""
        self.species_representatives_prev_gen.clear()
        for spec in self.species:
            if spec and spec.representative: 
                # зберігаємо копію, щоб мутації в наступному поколінні не вплинули на старого представника
                self.species_representatives_prev_gen[spec.id] = spec.representative.copy()
    def _get_next_genome_id(self) -> int:
        """Повертає наступний унікальний ID геному."""
        return next(self._genome_id_counter)

    def _create_initial_population(self, innovation_manager: InnovationManager) -> list[Genome]: 
        """Створює початкову популяцію геномів."""
        population = []
        for _ in range(self.population_size):
            genome_id = self._get_next_genome_id()
            # Передаємо той самий менеджер інновацій кожному новому геному
            genome = Genome(genome_id, self.num_inputs, self.num_outputs, self.config, innovation_manager)
            population.append(genome)
        # Після створення всієї популяції, лічильник інновацій в менеджері буде актуальним
        #print(f"Initial population created. Next innovation number: {innovation_manager.innovation_counter}")
        return population

    def _get_next_genome_id(self) -> int: 
        """Повертає наступний унікальний ID геному."""
        return next(self._genome_id_counter)
    
    def get_state_data(self) -> dict:
        """Збирає дані для збереження стану NEAT, забезпечуючи консистентність."""
        #print("DEBUG SAVE: Entered get_state_data.")
        
        relevant_genomes_map = {} # Використовуємо словник для унікальності за ID

        # 1. Геноми з поточної популяції (це популяція для НАСТУПНОЇ генерації, P_N+1)
        for g in self.population: 
            if g and g.id not in relevant_genomes_map:
                relevant_genomes_map[g.id] = g

        # 2. Геноми з членів поточних видів (це види S_N, їх члени - з популяції P_N)
        for spec in self.species:
            if spec:
                if spec.representative and spec.representative.id not in relevant_genomes_map:
                    relevant_genomes_map[spec.representative.id] = spec.representative
                for member in spec.members:
                    if member and member.id not in relevant_genomes_map:
                        relevant_genomes_map[member.id] = member
        
        # 3. Геноми з представників попереднього покоління (використовувались для видоутворення S_N, самі з P_N-1)
        for rep_genome in self.species_representatives_prev_gen.values():
            if rep_genome and rep_genome.id not in relevant_genomes_map:
                relevant_genomes_map[rep_genome.id] = rep_genome

        # 4. Найкращий геном за весь час
        if self.best_genome_overall and self.best_genome_overall.id not in relevant_genomes_map:
            relevant_genomes_map[self.best_genome_overall.id] = self.best_genome_overall
        
        # Зберігаємо копії всіх цих релевантних геномів
        all_referenced_genomes_copies = [g.copy() for g in relevant_genomes_map.values() if g]
        #print(f"DEBUG SAVE: Total unique relevant genomes to save in 'population_genomes': {len(all_referenced_genomes_copies)}")

        # Дані про види (species_state_data) беруться з поточного self.species (S_N)
        species_state_data_to_save = [spec.get_state_data() for spec in self.species if spec]
        
        # ID представників попереднього покоління (для видоутворення S_N)
        prev_gen_reps_data_ids = {
            spec_id: rep_g.id 
            for spec_id, rep_g in self.species_representatives_prev_gen.items() 
            if rep_g
        }

        # Зберігаємо ID поточної популяції (P_N+1), щоб знати, яку популяцію активувати після завантаження
        current_active_population_ids = [g.id for g in self.population if g]

        state = {
            'generation': self.generation, 
            'best_genome_overall': self.best_genome_overall.copy() if self.best_genome_overall else None,
            'innovation_manager_state': {
                '_next_node_id': self.innovation_manager._next_node_id,
                '_next_innovation_num': self.innovation_manager._next_innovation_num,
            },
            'population_genomes': all_referenced_genomes_copies, 
            'current_active_population_ids': current_active_population_ids,
            'species_state_data': species_state_data_to_save, 
            '_genome_id_counter_val': next(self._genome_id_counter) - 1,
            '_max_used_species_id': max((s.id for s in self.species if s), default=0),
            'species_representatives_prev_gen_ids': prev_gen_reps_data_ids,
            'first_goal_achieved_generation': self.first_goal_achieved_generation, 
            'generation_statistics': self.generation_statistics 
        }
        #print(f"DEBUG SAVE: Max species ID being saved: {state['_max_used_species_id']}")
        #print(f"DEBUG SAVE: Genome counter value being saved: {state['_genome_id_counter_val']}")
        #print(f"DEBUG SAVE: Saving {len(state['population_genomes'])} total genomes, {len(state['current_active_population_ids'])} active population genomes.")
        return state

    
    @classmethod
    def load_from_state_data(cls, state_data: dict, config: dict, num_inputs: int, num_outputs: int) -> 'NeatAlgorithm':
        initial_genome_id_to_set = state_data.get('_genome_id_counter_val', -1) + 1
        neat = cls(config, num_inputs, num_outputs, initial_genome_id_start=initial_genome_id_to_set, _is_loading=True)

        neat.generation = state_data['generation']
        
        best_genome_overall_data = state_data.get('best_genome_overall')
        if best_genome_overall_data:
            neat.best_genome_overall = best_genome_overall_data
        else:
            neat.best_genome_overall = None

        im_state = state_data['innovation_manager_state']
        neat.innovation_manager._next_node_id = im_state['_next_node_id']
        neat.innovation_manager._next_innovation_num = im_state['_next_innovation_num']
        neat.innovation_manager.reset_generation_history()

        # Завантажуємо ВСІ збережені геноми в загальну карту
        all_loaded_genomes_list = state_data.get('population_genomes', [])
        genomes_by_id = {genome.id: genome for genome in all_loaded_genomes_list if genome}
        #print(f"Info (Load): Loaded {len(all_loaded_genomes_list)} total genomes into master list. Genomes by ID map created with {len(genomes_by_id)} entries.")

        # Відновлюємо АКТИВНУ популяцію (P_N+1)
        current_active_population_ids = state_data.get('current_active_population_ids', [])
        neat.population = [genomes_by_id[gid] for gid in current_active_population_ids if gid in genomes_by_id]
        #print(f"Info (Load): Reconstructed active population with {len(neat.population)} genomes.")


        max_loaded_species_id = state_data.get('_max_used_species_id', 0)
        Species._species_counter = itertools.count(max_loaded_species_id + 1)
        #print(f"Info (Load): Species ID counter reset to start from {max_loaded_species_id + 1}.")
        
        neat.species = []
        loaded_species_data = state_data.get('species_state_data', [])
        #print(f"Info (Load): Attempting to load {len(loaded_species_data)} species records.")

        for s_data in loaded_species_data:
            species_id_from_data = s_data.get('id', 'Unknown_ID')
            representative_genome_obj = None
            rep_id = s_data.get('representative_id')
            member_ids_from_data = s_data.get('member_ids', [])
            
            # print(f"Debug (Load): Processing species data for ID {species_id_from_data}. Rep ID: {rep_id}. Member IDs: {member_ids_from_data}")

            if rep_id is not None:
                representative_genome_obj = genomes_by_id.get(rep_id)
                if not representative_genome_obj:
                    print(f"Warning (Load): Representative genome with ID '{rep_id}' for species '{species_id_from_data}' not found in loaded master genomes list.")

            if not representative_genome_obj and member_ids_from_data:
                # print(f"Info (Load): Rep ID '{rep_id}' for species '{species_id_from_data}' not found or was None. Attempting to find representative from its members.")
                for m_id_fallback in member_ids_from_data:
                    fallback_rep_obj = genomes_by_id.get(m_id_fallback)
                    if fallback_rep_obj:
                        representative_genome_obj = fallback_rep_obj
                        print(f"Info (Load): Using member ID '{m_id_fallback}' as representative for species '{species_id_from_data}'.")
                        break 
            
            if not representative_genome_obj:
                 print(f"Error (Load): Could not assign a representative for species ID '{species_id_from_data}'. Skipping this species.")
                 continue

            species_obj = Species(representative_genome_obj) 
            species_obj.id = species_id_from_data 
            species_obj.generations_since_improvement = s_data.get('generations_since_improvement', 0)
            species_obj.best_fitness_ever = s_data.get('best_fitness_ever', 0.0)
            species_obj.offspring_count = s_data.get('offspring_count', 0)
            
            species_obj.clear_members() 
            
            actually_added_members_count = 0
            for member_id in member_ids_from_data:
                member_genome = genomes_by_id.get(member_id)
                if member_genome:
                    species_obj.add_member(member_genome) 
                    actually_added_members_count += 1
            if actually_added_members_count > 0:
                neat.species.append(species_obj)
            
        neat.species_representatives_prev_gen = {}
        prev_gen_reps_ids_data = state_data.get('species_representatives_prev_gen_ids', {})
        for spec_id, rep_id_prev in prev_gen_reps_ids_data.items():
            rep_genome_prev = genomes_by_id.get(rep_id_prev)
            if rep_genome_prev:
                neat.species_representatives_prev_gen[spec_id] = rep_genome_prev
            
        neat.first_goal_achieved_generation = state_data.get('first_goal_achieved_generation')
        neat.generation_statistics = state_data.get('generation_statistics', [])
        print(f"NEAT state loaded. Gen: {neat.generation}, Pop: {len(neat.population)}, Species: {len(neat.species)}, PrevReps: {len(neat.species_representatives_prev_gen)}")
        return neat
    
    def _speciate_population(self):
        """
        Розподіляє геноми по видах на основі генетичної відстані,
        використовуючи представників з ПОПЕРЕДНЬОГО покоління.
        """
        threshold = self.config['COMPATIBILITY_THRESHOLD']
        c1 = self.config['C1_EXCESS']
        c2 = self.config['C2_DISJOINT']
        c3 = self.config['C3_WEIGHT']

        old_species_map = {s.id: s for s in self.species}

        species_to_compare_against = []
        existing_species_for_repopulation = []

        if self.generation > 0 and self.species_representatives_prev_gen:
            for spec_id, rep_genome in self.species_representatives_prev_gen.items():
                current_gen_species_instance = Species(rep_genome.copy())
                current_gen_species_instance.id = spec_id
                # Переносимо історію зі старого об'єкта виду, якщо він існував
                if spec_id in old_species_map:
                    original_species_history = old_species_map[spec_id]
                    current_gen_species_instance.generations_since_improvement = original_species_history.generations_since_improvement
                    current_gen_species_instance.best_fitness_ever = original_species_history.best_fitness_ever
                existing_species_for_repopulation.append(current_gen_species_instance)
            
            species_to_compare_against = existing_species_for_repopulation
        else:
            species_to_compare_against = []
            if not self.species and self.population:
                 print("Speciation: Initial speciation, no previous representatives.")
            elif self.species: 
                 for spec in self.species:
                     if spec.representative:
                         spec.clear_members() # Очищаємо для нового наповнення
                         species_to_compare_against.append(spec)
                 print(f"Speciation: Using {len(species_to_compare_against)} current representatives (e.g., after load).")


        newly_created_species_this_gen = []
        final_species_list = list(species_to_compare_against) 
        for genome in self.population:
            if not genome: continue

            assigned_to_existing_or_repopulating = False
            for spec_to_check in species_to_compare_against: 
                if not spec_to_check.representative: continue
                distance = genome.distance(spec_to_check.representative, c1, c2, c3)
                if distance < threshold:
                    spec_to_check.add_member(genome)
                    assigned_to_existing_or_repopulating = True
                    break
            
            if not assigned_to_existing_or_repopulating:
                assigned_to_newly_created = False
                for new_spec in newly_created_species_this_gen:
                    distance = genome.distance(new_spec.representative, c1, c2, c3)
                    if distance < threshold:
                        new_spec.add_member(genome)
                        assigned_to_newly_created = True
                        break
                if not assigned_to_newly_created:
                    brand_new_species = Species(genome) 
                    newly_created_species_this_gen.append(brand_new_species)
                    final_species_list.append(brand_new_species)

        self.species = [s for s in final_species_list if s.members]
        # Оновлюємо представників для НАСТУПНОГО покоління
        for spec in self.species:
            spec.update_representative()

    def _calculate_adjusted_fitness(self):
        for spec in self.species:
            spec.calculate_adjusted_fitness_and_sum()

    def _determine_num_offspring(self) -> dict[int, int]:
        num_offspring_map = {}
        total_adjusted_fitness_sum = sum(spec.total_adjusted_fitness for spec in self.species if spec.total_adjusted_fitness > 0)

        if total_adjusted_fitness_sum <= 0:
            num_active_species = len(self.species)
            if not num_active_species: return {}
            base_offspring = self.population_size // num_active_species
            remainder = self.population_size % num_active_species
            active_species_list = [spec for spec in self.species] 
            for i, spec in enumerate(active_species_list):
                 count = base_offspring + (1 if i < remainder else 0)
                 num_offspring_map[spec.id] = count
                 spec.offspring_count = count 
            return num_offspring_map

        total_spawn = 0
        spawn_fractions = {}
        for spec in self.species:
             if spec.total_adjusted_fitness > 0:
                 proportion = spec.total_adjusted_fitness / total_adjusted_fitness_sum
                 spawn_amount_float = proportion * self.population_size
                 num_offspring_map[spec.id] = int(spawn_amount_float)
                 spawn_fractions[spec.id] = spawn_amount_float - int(spawn_amount_float)
                 total_spawn += int(spawn_amount_float)
             else:
                  num_offspring_map[spec.id] = 0
             spec.offspring_count = num_offspring_map.get(spec.id, 0) 

        spawn_diff = self.population_size - total_spawn
        if spawn_diff > 0:
             sorted_species_by_fraction = sorted(spawn_fractions.items(), key=lambda item: item[1], reverse=True)
             for i in range(min(spawn_diff, len(sorted_species_by_fraction))):
                 spec_id_to_increment = sorted_species_by_fraction[i][0]
                 num_offspring_map[spec_id_to_increment] += 1
                 spec = next((s for s in self.species if s.id == spec_id_to_increment), None)
                 if spec: spec.offspring_count += 1

        return num_offspring_map


    def _handle_stagnation(self):
         species_to_keep = []
         max_stagnation = self.config.get('MAX_STAGNATION', 15)
         num_non_stagnant = 0
         for spec in self.species:
             spec.update_stagnation_and_best_fitness()
             if spec.generations_since_improvement <= max_stagnation:
                  num_non_stagnant += 1

         can_remove_stagnant = num_non_stagnant >= 2 
         kept_species_ids = set()
         if not self.species: return 
         # Завжди зберігаємо найкращий вид, навіть якщо він стагнує
         best_overall_species = max(self.species, key=lambda s: s.best_fitness_ever, default=None)

         for spec in self.species:
             is_stagnant = spec.generations_since_improvement > max_stagnation
             should_keep = False
             if spec == best_overall_species:
                 should_keep = True
             elif not is_stagnant: 
                  should_keep = True
             elif not can_remove_stagnant: 
                  should_keep = True

             if should_keep:
                  species_to_keep.append(spec)
                  kept_species_ids.add(spec.id)
             else:
                  print(f"Species {spec.id} removed due to stagnation ({spec.generations_since_improvement} gens).")

         self.species = species_to_keep
         print(f"Species after stagnation handling: {[s.id for s in self.species]}")



    def _reproduce(self) -> list[Genome]:
        next_population = []
        survival_threshold = self.config.get('SELECTION_PERCENTAGE', 0.2)
        prob_crossover = self.config.get('CROSSOVER_RATE', 0.75)
        elitism_count = self.config.get('ELITISM', 1)

        self._handle_stagnation() # забираємо стагнуючі види з пулу для розмноження
        num_offspring_map = self._determine_num_offspring() 

        if not self.species:
             print("Error: No species left to reproduce. Resetting population.")
             return self._create_initial_population(self.innovation_manager)
        
        all_parents_list = []
        for spec in self.species:
             all_parents_list.extend(spec.members)

        for spec in self.species:
            spawn_amount = spec.offspring_count
            if spawn_amount == 0 or not spec.members: continue

            spec.sort_members_by_fitness()
            elites_added = 0
            if elitism_count > 0:
                for i in range(min(elitism_count, spawn_amount, len(spec.members))):
                     elite_copy = spec.members[i].copy()
                     #Елітам теж треба дати новий ID
                     elite_copy.id = self._get_next_genome_id()
                     next_population.append(elite_copy)
                     elites_added += 1
            spawn_amount -= elites_added

            if spawn_amount == 0: continue

            parents = spec.get_parents(survival_threshold)
            if not parents: parents = spec.members[:1] # Якщо всі погані, беремо найкращого
            if not parents: continue # Якщо вид порожній (не має статися)

            while spawn_amount > 0:
                 parent1 = random.choice(parents)
                 child = None
                 if random.random() < prob_crossover and len(parents) > 1:
                     parent2 = random.choice(parents)
                     attempts = 0
                     while parent2.id == parent1.id and attempts < 5 and len(parents) > 1:
                         parent2 = random.choice(parents)
                         attempts += 1
                     g1_is_fitter = parent1.fitness > parent2.fitness
                     child = self._crossover_with_innovation_manager(parent1, parent2, g1_is_fitter)
                 else:
                     child = parent1.copy()

                 child.id = self._get_next_genome_id()
                 # Мутації
                 child.mutate_weights()
                 if random.random() < self.config['ADD_CONNECTION_RATE']:
                     child.mutate_add_connection(self.innovation_manager)
                 if random.random() < self.config['ADD_NODE_RATE']:
                     child.mutate_add_node(self.innovation_manager)

                 next_population.append(child)
                 spawn_amount -= 1

        current_pop_size = len(next_population)
        if current_pop_size < self.population_size:
             all_parents_list.sort(key=lambda g: g.fitness, reverse=True) # Сортуємо всіх батьків
             needed = self.population_size - current_pop_size
             fill_idx = 0
             while needed > 0 and all_parents_list:
                 filler_parent = all_parents_list[fill_idx % len(all_parents_list)]
                 if filler_parent:
                     filler = filler_parent.copy()
                     filler.id = self._get_next_genome_id()
                     filler.mutate_weights()
                     next_population.append(filler)
                     needed -= 1
                 fill_idx += 1
             # Якщо батьків не вистачило, створюємо повністю нові (малоймовірно)
             while len(next_population) < self.population_size:
                  new_genome = Genome(self._get_next_genome_id(), self.num_inputs, self.num_outputs, self.config, self.innovation_manager)
                  next_population.append(new_genome)


        elif current_pop_size > self.population_size:
             next_population = next_population[:self.population_size]


        return next_population

    def _crossover_with_innovation_manager(self, parent1: Genome, parent2: Genome, g1_is_fitter: bool) -> Genome:
        """Виконує кросовер з правильним innovation_manager."""
        parent1._innovation_manager = self.innovation_manager
        parent2._innovation_manager = self.innovation_manager
        
        child = Genome.crossover(parent1, parent2, g1_is_fitter)
        
        if hasattr(parent1, '_innovation_manager'):
            delattr(parent1, '_innovation_manager')
        if hasattr(parent2, '_innovation_manager'):
            delattr(parent2, '_innovation_manager')
        
        return child
    def run_generation(self, evaluation_function):
        """Запускає один цикл покоління з паралельною оцінкою."""
        self.generation += 1
        self.innovation_manager.reset_generation_history()

        total_fitness = 0.0
        max_fitness = -float('inf')
        current_best_genome = None
        evaluation_results_with_goal_flag = {} 

        # Паралельна оцінка фітнесу 
        population_to_evaluate = [(g.id, g) for g in self.population if g] # Список кортежів (id, genome)
        num_processes = self.config.get('NUM_PROCESSES', os.cpu_count()) # Кількість процесів

        print(f"Starting parallel evaluation for {len(population_to_evaluate)} genomes using {num_processes} processes...")
        futures = {}
        # щоб уникнути проблем із спільним доступом
        config_copy = self.config.copy()
        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Надсилаємо завдання: evaluate_single_genome(genome_tuple, config)
                for genome_tuple in population_to_evaluate:
                    future = executor.submit(evaluation_function, genome_tuple, config_copy)
                    # Зберігаємо future та відповідний ID геному
                    futures[future] = genome_tuple[0] # Ключ - future, значення - genome_id

                # Збираємо результати по мірі завершення
                for future in as_completed(futures):
                    genome_id = futures[future]
                    try:
                        _, fitness, reached_goal_flag = future.result() 
                        evaluation_results_with_goal_flag[genome_id] = (fitness, reached_goal_flag)
                    except Exception as exc:
                        print(f'Genome {genome_id} evaluation generated an exception: {exc}')
                        evaluation_results_with_goal_flag[genome_id] = (0.001, False) 
        except Exception as pool_exc:
             print(f"Error during ProcessPoolExecutor execution: {pool_exc}")
             for genome_id, genome_obj in population_to_evaluate: 
                 try:
                     _, fitness, reached_goal_flag = evaluation_function((genome_id, genome_obj), config_copy)
                     evaluation_results_with_goal_flag[genome_id] = (fitness, reached_goal_flag)
                 except Exception as eval_exc:
                      print(f"Sequential evaluation error for genome {genome_id}: {eval_exc}")
                      evaluation_results_with_goal_flag[genome_id] = (0.001, False)
        
        any_genome_reached_goal_this_gen = False 
        valid_evaluated_genomes = 0
        for genome in self.population:
             if genome:
                  fitness, reached_goal_flag = evaluation_results_with_goal_flag.get(genome.id, (0.001, False))
                  genome.fitness = fitness
                  if reached_goal_flag: 
                      any_genome_reached_goal_this_gen = True
                  
                  total_fitness += genome.fitness
                  valid_evaluated_genomes += 1
                  if genome.fitness > max_fitness:
                       max_fitness = genome.fitness
                       current_best_genome = genome

        avg_fitness = total_fitness / valid_evaluated_genomes if valid_evaluated_genomes > 0 else 0.0

        if self.best_genome_overall is None or (current_best_genome and current_best_genome.fitness > self.best_genome_overall.fitness):
            self.best_genome_overall = current_best_genome.copy() if current_best_genome else None

        if any_genome_reached_goal_this_gen and self.first_goal_achieved_generation is None:
            self.first_goal_achieved_generation = self.generation
            print(f"INFO: Goal first achieved at generation {self.generation}!")

        stats = {
            "generation": self.generation,
            "max_fitness": max_fitness if max_fitness > -float('inf') else None,
            "average_fitness": avg_fitness,
            "num_species": len(self.species), 
            "best_genome_current_gen": current_best_genome,
            "best_genome_overall": self.best_genome_overall,
            "first_goal_achieved_generation": self.first_goal_achieved_generation
        }
        self._update_previous_gen_representatives() 
        self._speciate_population()
        for spec in self.species:
            if spec.members:
                spec.sort_members_by_fitness()
        self._calculate_adjusted_fitness()
        next_population = self._reproduce()
        self.population = next_population
        stats["num_species_after_speciation"] = len(self.species) 
        self.generation_statistics.append(stats) # Зберігаємо фінальну статистик
        return stats

    def get_best_genome_overall(self) -> Genome | None:
        return self.best_genome_overall