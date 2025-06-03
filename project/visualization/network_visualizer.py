import math
import os
from typing import Dict, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import traceback

try:
    import sys
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(viz_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    from neat.genome import Genome
except ImportError as e:
    print(f"Error importing from neat package in network_visualizer.py: {e}")
    class Genome:
        def __init__(self): self.nodes = {}; self.connections = {}; self.config = {}
        def get_input_output_bias_ids(self): return [], [], None
    class NodeGene: pass
    class ConnectionGene: pass

BASE_NODE_RADIUS = 12 # Базовий радіус
BASE_LAYER_SPACING_X = 120
BASE_NODE_SPACING_Y = 45
BASE_IMAGE_PADDING_TOP_BOTTOM = 30
BASE_IMAGE_PADDING_LEFT = 100
BASE_IMAGE_PADDING_RIGHT = 100
BASE_LABEL_OFFSET_X_INPUT = -15
BASE_LABEL_OFFSET_X_OUTPUT = 15
BASE_MAX_LINE_WIDTH = 4
BASE_MIN_LINE_WIDTH = 1
BASE_NODE_FONT_SIZE = 11
BASE_LABEL_FONT_SIZE = 12 
# Кольори
COLOR_INPUT = "#87CEEB"
COLOR_OUTPUT = "#FFA07A"
COLOR_HIDDEN = "#D3D3D3"
COLOR_BIAS = "#FFFACD"
COLOR_NODE_OUTLINE = "#333333"
COLOR_CONN_POSITIVE = "#4CAF50"
COLOR_CONN_NEGATIVE = "#F44336"
COLOR_CONN_DISABLED = "#BDBDBD"
COLOR_BACKGROUND = "#282c34"
COLOR_NODE_TEXT = "black"
COLOR_LABEL_TEXT = "white"
MAX_LINE_WIDTH = 6 
MIN_LINE_WIDTH = 1

def get_font(base_size: int, zoom: float) -> ImageFont.ImageFont:
    """Завантажує шрифт заданого розміру."""
    scaled_size = max(8, int(base_size * zoom)) 
    try:
        try: return ImageFont.truetype("consola.ttf", scaled_size)
        except IOError: return ImageFont.truetype("cour.ttf", scaled_size)
    except IOError:
        print(f"Warning: Consolas/Courier New font not found for size {scaled_size}. Using default PIL font.")
        try: return ImageFont.truetype("arial.ttf", scaled_size)
        except: return ImageFont.load_default()

def _get_node_color(node_type: str) -> str:
    """Повертає колір вузла за його типом."""
    type_upper = node_type.upper()
    if type_upper == "INPUT": return COLOR_INPUT
    if type_upper == "OUTPUT": return COLOR_OUTPUT
    if type_upper == "HIDDEN": return COLOR_HIDDEN
    if type_upper == "BIAS": return COLOR_BIAS
    return "grey"

def _assign_layers_simple(genome: Genome) -> Tuple[Dict[int, int], Dict[int, List[int]], int]:
    """
    Спрощене призначення шарів: 0=Input/Bias, 1=Hidden, 2=Output.
    Не враховує глибину прихованих вузлів.
    """
    layers = {}
    node_ids_in_layers: Dict[int, List[int]] = {0: [], 1: [], 2: []} # Input, Hidden, Output
    nodes_in_genome = set(genome.nodes.keys())
    input_ids, output_ids, bias_id = genome.get_input_output_bias_ids()
    max_layer = 0

    # Шар 0: Вхідні та Біас
    for nid in input_ids:
        if nid in nodes_in_genome:
            layers[nid] = 0
            node_ids_in_layers[0].append(nid)
    if bias_id is not None and bias_id in nodes_in_genome:
        layers[bias_id] = 0
        node_ids_in_layers[0].append(bias_id)

    # Шар 1: Приховані
    has_hidden = False
    for nid, node in genome.nodes.items():
        if node.type == "HIDDEN":
            layers[nid] = 1
            node_ids_in_layers[1].append(nid)
            has_hidden = True

    # Шар 2 : Вихідні
    output_layer = 2 if has_hidden else 1
    max_layer = output_layer
    for nid in output_ids:
        if nid in nodes_in_genome:
            layers[nid] = output_layer
            node_ids_in_layers[output_layer].append(nid)

    # Видаляємо порожні шари та сортуємо
    final_node_ids_in_layers = {}
    sorted_layer_keys = sorted(node_ids_in_layers.keys())
    current_layer_index = 0
    final_max_layer_index = 0
    for key in sorted_layer_keys:
        if node_ids_in_layers[key]:
             node_ids = sorted(node_ids_in_layers[key])
             final_node_ids_in_layers[current_layer_index] = node_ids
             for nid in node_ids:
                 layers[nid] = current_layer_index
             final_max_layer_index = current_layer_index
             current_layer_index += 1


    return layers, final_node_ids_in_layers, final_max_layer_index

def _calculate_base_node_positions(genome: Genome, node_ids_in_layers: Dict[int, List[int]], max_layer: int) -> Tuple[Dict[int, Tuple[int, int]], int, int]:
    """
    Розраховує БАЗОВІ позиції вузлів (без зуму) та БАЗОВІ розміри області вузлів.
    """
    positions = {}
    max_nodes_in_layer = 0
    layer_keys = sorted(node_ids_in_layers.keys())

    for layer_num in layer_keys:
        layer_nodes = node_ids_in_layers.get(layer_num, [])
        max_nodes_in_layer = max(max_nodes_in_layer, len(layer_nodes))

    base_node_area_width = max(0, max_layer) * BASE_LAYER_SPACING_X 
    base_node_area_height = max(BASE_NODE_RADIUS * 2, (max_nodes_in_layer - 1) * BASE_NODE_SPACING_Y) if max_nodes_in_layer > 0 else 0

    for layer_num in layer_keys:
        layer_nodes = node_ids_in_layers.get(layer_num, [])
        if not layer_nodes: continue

        num_nodes = len(layer_nodes)
        layer_x = layer_num * BASE_LAYER_SPACING_X

        total_layer_height = max(0, num_nodes - 1) * BASE_NODE_SPACING_Y
        start_y = (base_node_area_height - total_layer_height) / 2 

        for i, node_id in enumerate(layer_nodes):
            node_y = start_y + i * BASE_NODE_SPACING_Y
            positions[node_id] = (int(layer_x), int(node_y))

    return positions, int(base_node_area_width), int(base_node_area_height)

def _calculate_base_node_positions(genome: Genome, node_ids_in_layers: Dict[int, List[int]], max_layer: int) -> Tuple[Dict[int, Tuple[int, int]], int, int]:
    """
    Розраховує БАЗОВІ позиції вузлів (без зуму) та БАЗОВІ розміри області вузлів.
    """
    positions = {}
    max_nodes_in_layer = 0
    layer_keys = sorted(node_ids_in_layers.keys())

    for layer_num in layer_keys:
        layer_nodes = node_ids_in_layers.get(layer_num, [])
        max_nodes_in_layer = max(max_nodes_in_layer, len(layer_nodes))

    base_node_area_width = max(0, max_layer) * BASE_LAYER_SPACING_X 
    base_node_area_height = max(BASE_NODE_RADIUS * 2, (max_nodes_in_layer - 1) * BASE_NODE_SPACING_Y) if max_nodes_in_layer > 0 else 0

    # Розраховуємо базові позиції відносно (0, 0) верхнього лівого кута області вузлів
    for layer_num in layer_keys:
        layer_nodes = node_ids_in_layers.get(layer_num, [])
        if not layer_nodes: continue

        num_nodes = len(layer_nodes)
        layer_x = layer_num * BASE_LAYER_SPACING_X

        total_layer_height = max(0, num_nodes - 1) * BASE_NODE_SPACING_Y
        start_y = (base_node_area_height - total_layer_height) / 2 # Центрування відносно області

        for i, node_id in enumerate(layer_nodes):
            node_y = start_y + i * BASE_NODE_SPACING_Y
            positions[node_id] = (int(layer_x), int(node_y))

    return positions, int(base_node_area_width), int(base_node_area_height)


def _get_node_labels(genome: Genome) -> Dict[int, str]:
    """Створює словник підписів для вхідних/вихідних вузлів."""
    labels = {}
    config = genome.config if hasattr(genome, 'config') and isinstance(genome.config, dict) else {}
    inp_ids, out_ids, bias_id = genome.get_input_output_bias_ids()
    inp_ids_sorted = sorted(inp_ids)
    out_ids_sorted = sorted(out_ids)

    sensor_names = []
    sensor_names.extend([f"Rng{i}" for i in range(config.get('NUM_RANGEFINDERS', 8))])
    sensor_names.extend([f"Rad{i}" for i in range(config.get('NUM_RADAR_SLICES', 8))])
    sensor_names.extend(["HeadX", "HeadY", "Vel"])

    for i, nid in enumerate(inp_ids_sorted):
        if nid in genome.nodes:
             labels[nid] = sensor_names[i] if i < len(sensor_names) else f"IN_{i}"

    if bias_id is not None and bias_id in genome.nodes:
         labels[bias_id] = "Bias"

    output_action_names = ["TrnL", "TrnR", "Accel", "Brake"]
    for i, nid in enumerate(out_ids_sorted):
        if nid in genome.nodes:
             labels[nid] = output_action_names[i] if i < len(output_action_names) else f"OUT_{i}"

    return labels

def visualize_network(genome: 'Genome', zoom_factor: float = 1.0) -> Optional[Image.Image]:
    """
    Створює зображення фенотипу нейронної мережі з геному з урахуванням зуму.
    Автоматично розраховує розмір зображення.

    Args:
        genome (Genome): Геном для візуалізації.
        zoom_factor (float): Коефіцієнт масштабування (1.0 = базовий).

    Returns:
        Optional[Image.Image]: PIL Image об'єкт або None при помилці.
    """
    if not isinstance(genome, Genome) or not genome.nodes:
        print("Warning: Invalid or empty genome passed to visualize_network.")
        return None
    zoom_factor = max(0.1, zoom_factor) 

    try:
        node_radius = int(BASE_NODE_RADIUS * zoom_factor)
        padding_top_bottom = int(BASE_IMAGE_PADDING_TOP_BOTTOM * zoom_factor)
        padding_left = int(BASE_IMAGE_PADDING_LEFT * zoom_factor) 
        padding_right = int(BASE_IMAGE_PADDING_RIGHT * zoom_factor)
        label_offset_x_input = int(BASE_LABEL_OFFSET_X_INPUT * zoom_factor)
        label_offset_x_output = int(BASE_LABEL_OFFSET_X_OUTPUT * zoom_factor)
        max_line_width = max(1, int(BASE_MAX_LINE_WIDTH * zoom_factor))
        min_line_width = max(1, int(BASE_MIN_LINE_WIDTH * zoom_factor))

        node_font = get_font(BASE_NODE_FONT_SIZE, zoom_factor)
        label_font = get_font(BASE_LABEL_FONT_SIZE, zoom_factor)

        layers, node_ids_in_layers, max_layer = _assign_layers_simple(genome)
        if not layers: return None

        base_positions, base_node_area_width, base_node_area_height = _calculate_base_node_positions(
            genome, node_ids_in_layers, max_layer
        )
        if not base_positions: return None

        img_width = int(base_node_area_width * zoom_factor) + padding_left + padding_right
        img_height = int(base_node_area_height * zoom_factor) + padding_top_bottom * 2
        img_width = max(300, img_width) 
        img_height = max(250, img_height)

        final_positions = {}
        offset_x = padding_left
        offset_y = padding_top_bottom + (img_height - 2 * padding_top_bottom - base_node_area_height * zoom_factor) / 2

        for node_id, (base_x, base_y) in base_positions.items():
             final_x = offset_x + base_x * zoom_factor
             final_y = offset_y + base_y * zoom_factor
             final_positions[node_id] = (int(final_x), int(final_y))

        image = Image.new("RGB", (img_width, img_height), COLOR_BACKGROUND)
        draw = ImageDraw.Draw(image)

        max_abs_weight = 1e-6
        for conn in genome.connections.values():
            if conn.enabled:
                max_abs_weight = max(max_abs_weight, abs(conn.weight))

        for conn in genome.connections.values():
            if conn.in_node_id in final_positions and conn.out_node_id in final_positions:
                start_pos = final_positions[conn.in_node_id]
                end_pos = final_positions[conn.out_node_id]

                if conn.enabled:
                    weight = conn.weight
                    line_width_ratio = min(1.0, abs(weight) / max_abs_weight)
                    line_width = int(min_line_width + math.sqrt(line_width_ratio) * (max_line_width - min_line_width))
                    line_width = max(1, line_width)
                    intensity_ratio = 0.5 + 0.5 * line_width_ratio 

                    if weight > 0:
                        r, g, b = 76, 175, 80
                        final_color = (int(r*intensity_ratio), int(g*intensity_ratio), int(b*intensity_ratio))
                    else:
                        r, g, b = 244, 67, 54
                        final_color = (int(r*intensity_ratio), int(g*intensity_ratio), int(b*intensity_ratio))
                    color_hex = f"#{final_color[0]:02x}{final_color[1]:02x}{final_color[2]:02x}"
                else:
                    color_hex = COLOR_CONN_DISABLED
                    line_width = 1

                draw.line([start_pos, end_pos], fill=color_hex, width=line_width)

        node_labels = _get_node_labels(genome)
        for node_id, pos in final_positions.items():
            if node_id not in genome.nodes: continue
            node = genome.nodes[node_id]
            color = _get_node_color(node.type)
            x, y = pos

            draw.ellipse(
                (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
                fill=color, outline=COLOR_NODE_OUTLINE, width=max(1, int(2 * zoom_factor))
            )
            text = str(node.id)
            try: draw.text((x, y), text, fill=COLOR_NODE_TEXT, font=node_font, anchor="mm")
            except AttributeError: 
                bbox = draw.textbbox((0, 0), text, font=node_font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                draw.text((x - tw / 2, y - th / 2), text, fill=COLOR_NODE_TEXT, font=node_font)

            label_text = node_labels.get(node_id)
            if label_text:
                label_color = COLOR_LABEL_TEXT
                anchor = "mm"
                if node.type == "INPUT" or node.type == "BIAS":
                    label_x = x + label_offset_x_input
                    anchor = "rm"
                elif node.type == "OUTPUT":
                    label_x = x + label_offset_x_output
                    anchor = "lm"
                else: label_text = None 

                if label_text:
                    try: draw.text((label_x, y), label_text, fill=label_color, font=label_font, anchor=anchor)
                    except TypeError: 
                        align = "right" if anchor == "rm" else "left"
                        lbbox = draw.textbbox((0,0), label_text, font=label_font)
                        ly = y - (lbbox[3]-lbbox[1])/2
                        draw.text((label_x, ly), label_text, fill=label_color, font=label_font, align=align)

        return image

    except Exception as e:
        print(f"ERROR during network visualization: {e}")
        traceback.print_exc()
        try:
            error_img = Image.new("RGB", (300, 200), "pink")
            draw_err = ImageDraw.Draw(error_img)
            draw_err.text((10, 10), f"Viz Error:\n{e}", fill="black")
            return error_img
        except: return None

