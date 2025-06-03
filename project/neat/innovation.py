class InnovationManager:
    """
    Клас для відстеження глобальних інноваційних номерів для нових
    структурних мутацій (додавання вузлів та з'єднань) в NEAT.

    Забезпечує, щоб однакові структурні зміни, що виникають незалежно
    в одному поколінні, отримували той самий інноваційний номер.
   
    """
    def __init__(self, start_node_id=0, start_innovation_num=0):
        # Глобальні лічильники, які НЕ скидаються між поколіннями
        self._next_node_id = int(start_node_id)
        self._next_innovation_num = int(start_innovation_num)

        # Історія інновацій в ПОТОЧНОМУ поколінні. Скидається на початку кожного покоління.
        # Зберігає відображення для НОВИХ з'єднань: (in_node_id, out_node_id) -> innovation_num
        self._current_generation_connection_innovations = {}
        # Зберігає відображення для НОВИХ вузлів:
        # split_conn_innov -> (new_node_id, new_conn_innov1, new_conn_innov2)
        # де new_conn_innov1 - інновація зв'язку in -> new_node
        # де new_conn_innov2 - інновація зв'язку new_node -> out
        self._current_generation_node_innovations = {}

    def get_node_id(self) -> int:
        """Повертає наступний доступний унікальний ID для нового вузла."""
        new_id = self._next_node_id
        self._next_node_id += 1
        return new_id

    def get_connection_innovation(self, in_node_id: int, out_node_id: int) -> int:
        """
        Повертає або реєструє інноваційний номер для з'єднання.
        Якщо така інновація (пара in->out) вже зареєстрована в поточному
        поколінні, повертає існуючий номер. В іншому випадку, створює
        новий глобальний номер, реєструє його для поточного покоління
        і повертає новий номер.
        """
        key = (int(in_node_id), int(out_node_id))
        innovation_num = self._current_generation_connection_innovations.get(key)

        if innovation_num is not None:
            return innovation_num
        else:
            new_innov = self._next_innovation_num
            self._current_generation_connection_innovations[key] = new_innov
            self._next_innovation_num += 1
            return new_innov

    def register_node_addition(self, split_connection_innovation: int, in_node_id: int, out_node_id: int) -> tuple[int, int, int]:
        key = int(split_connection_innovation)
        existing_innovation_data = self._current_generation_node_innovations.get(key)

        if existing_innovation_data is not None:
             # Ця мутація вже відбулася
             return existing_innovation_data
        else:
             # Генеруємо ID для нового вузла
             new_node_id = self.get_node_id()
             # Генеруємо/отримуємо інновації для нових з'єднань
             new_conn1_innov = self.get_connection_innovation(in_node_id, new_node_id)
             new_conn2_innov = self.get_connection_innovation(new_node_id, out_node_id)

             innovation_data = (new_node_id, new_conn1_innov, new_conn2_innov)
             self._current_generation_node_innovations[key] = innovation_data
             return innovation_data

    def reset_generation_history(self):
        """
        Скидає історію інновацій ПОТОЧНОГО покоління.
        Глобальні лічильники (_next_node_id, _next_innovation_num) НЕ скидаються.
        """
        self._current_generation_connection_innovations.clear()
        self._current_generation_node_innovations.clear()
        # print("Generation innovation history reset.")

    @property
    def node_id_counter(self) -> int:
        """Поточне значення лічильника ID вузлів."""
        return self._next_node_id

    @property
    def innovation_counter(self) -> int:
        """Поточне значення лічильника інновацій з'єднань."""
        return self._next_innovation_num

    def __repr__(self):
        return (f"InnovationManager(next_node={self._next_node_id}, "
                f"next_innov={self._next_innovation_num}, "
                f"gen_conn_hist_size={len(self._current_generation_connection_innovations)}, "
                f"gen_node_hist_size={len(self._current_generation_node_innovations)})")