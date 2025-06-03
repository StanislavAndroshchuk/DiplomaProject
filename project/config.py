# --- Параметри популяції та еволюції ---
POPULATION_SIZE = 150
MAX_GENERATIONS = 1000
FITNESS_THRESHOLD = 2000.0
IS_REWARDING_EFFICIENCY = True

# --- Параметри видоутворення (Speciation) ---
C1_EXCESS = 1.0
C2_DISJOINT = 1.0
C3_WEIGHT = 0.9
COMPATIBILITY_THRESHOLD = 5.0
MAX_STAGNATION = 25

# --- Параметри мутацій ---
ADD_CONNECTION_RATE = 0.1
ADD_NODE_RATE = 0.05
WEIGHT_MUTATE_RATE = 0.7
WEIGHT_REPLACE_RATE = 0.1
WEIGHT_MUTATE_POWER = 0.5
WEIGHT_CAP = 8.0
WEIGHT_INIT_RANGE = 1.0


# --- Параметри початкової структури ---
INITIAL_CONNECTIONS = 8 # Кількість випадкових початкових з'єднань
NUM_OUTPUTS = 4 # [TurnL, TurnR, Brake, Accel] 
# --- Параметри кросоверу та відбору ---
CROSSOVER_RATE = 0.75
INHERIT_DISABLED_GENE_RATE = 0.75
ELITISM = 0
SELECTION_PERCENTAGE = 0.20

# --- Параметри середовища та симуляції ---
MAZE_WIDTH = 13
MAZE_HEIGHT = 13
# Перевірка на непарність 
if MAZE_WIDTH % 2 == 0: MAZE_WIDTH += 1
if MAZE_HEIGHT % 2 == 0: MAZE_HEIGHT += 1
MAZE_WIDTH = max(5, MAZE_WIDTH)
MAZE_HEIGHT = max(5, MAZE_HEIGHT)
MAZE_SEED = None
MAX_STEPS_PER_EVALUATION = 400

# --- Параметри агента ---
NUM_RANGEFINDERS = 4
RANGEFINDER_MAX_DIST = 8.0
NUM_RADAR_SLICES = 4
AGENT_MAX_SPEED = 1

# --- Параметри візуалізації ---
CELL_SIZE_PX = 20
INFO_PANEL_WIDTH_PX = 300

