import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from typing import List
import sys
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if project_root_path not in sys.path:
    sys.path.append(project_root_path)

try:
    from neat.data_analyzer import NEATDataAnalyzer
except ImportError as e:
    messagebox.showerror("Помилка імпорту")
    exit()

class MultiRunComparerApp:
    MAX_FILES = 5
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown']
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Порівняння запусків NEAT")
        master.geometry("1200x800")

        self.selected_files_paths: List[str] = []
        self.analyzers: List[NEATDataAnalyzer] = []

        top_frame = ttk.Frame(master, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(master, padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.select_button = ttk.Button(top_frame, text="Вибрати файли (до 5)", command=self._select_files)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.files_label_var = tk.StringVar(value="Файли не вибрано")
        files_display_label = ttk.Label(top_frame, textvariable=self.files_label_var, wraplength=700)
        files_display_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.plot_button = ttk.Button(top_frame, text="Побудувати графіки", command=self._plot_data, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True) 
        plt.subplots_adjust(hspace=0.4) 

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw() 

    def _select_files(self):
        files = filedialog.askopenfilenames(
            title=f"Виберіть до {self.MAX_FILES} JSON файлів",
            filetypes=[("JSON файли", "*.json"), ("Всі файли", "*.*")]
        )

        if not files:
            return

        if len(files) > self.MAX_FILES:
            messagebox.showwarning("Забагато файлів",
                                   f"Будь ласка, виберіть не більше {self.MAX_FILES} файлів. "
                                   f"Буде використано перші {self.MAX_FILES}.")
            files = files[:self.MAX_FILES]

        self.selected_files_paths = list(files)
        self.analyzers = []

        loaded_file_names = []
        for i, f_path in enumerate(self.selected_files_paths):
            try:
                analyzer = NEATDataAnalyzer(f_path)
                if analyzer.generation_history:
                    self.analyzers.append(analyzer)
                    loaded_file_names.append(f"{i+1}. {os.path.basename(f_path)}")
                else:
                    messagebox.showwarning("Порожній файл", f"Тест '{os.path.basename(f_path)}' не містить історії поколінь і буде проігноровано.")
            except Exception as e:
                messagebox.showerror("Помилка завантаження файлу",
                                     f"Не вдалося завантажити або обробити файл:\n{os.path.basename(f_path)}\n\nПомилка: {e}")

        if self.analyzers:
            self.files_label_var.set("Вибрані файли:\n" + "\n".join(loaded_file_names))
            self.plot_button.config(state=tk.NORMAL)
        else:
            self.files_label_var.set("Не вдалося завантажити жодного файлу з даними.")
            self.plot_button.config(state=tk.DISABLED)
            self._clear_plots()

    def _clear_plots(self):
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.axes[0].set_title("Максимальний фітнес за поколіннями", fontsize=10)
        self.axes[0].set_ylabel("Макс. фітнес", fontsize=9)

        self.axes[1].set_title("Середній фітнес за поколіннями", fontsize=10)
        self.axes[1].set_ylabel("Сер. фітнес", fontsize=9)

        self.axes[2].set_title("Кількість видів за поколіннями", fontsize=10)
        self.axes[2].set_ylabel("К-ть видів", fontsize=9)
        self.axes[2].set_xlabel("Покоління", fontsize=9)
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


    def _plot_data(self):
        if not self.analyzers:
            messagebox.showinfo("Немає даних", "Будь ласка, спочатку виберіть та завантажте файли.")
            return

        self._clear_plots()

        max_generations_overall = 0

        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.get_fitness_statistics()
            if df.empty or 'generation' not in df.columns:
                messagebox.showwarning("Відсутні дані",
                                       f"Тест '{os.path.basename(self.selected_files_paths[i])}' "
                                       f"не містить необхідних даних про покоління.")
                continue

            generations = df['generation']
            max_fitness = df.get('max_fitness', [])
            avg_fitness = df.get('average_fitness', [])
            num_species = df.get('num_species', [])

            if not generations.empty:
                 max_generations_overall = max(max_generations_overall, generations.max())


            label = f"Тест {i+1}" 
            color = self.COLORS[i % len(self.COLORS)]
            linestyle = self.LINESTYLES[i % len(self.LINESTYLES)]

            if not max_fitness.empty:
                self.axes[0].plot(generations, max_fitness, label=label, color=color, linestyle=linestyle, linewidth=1.5)

            if not avg_fitness.empty:
                self.axes[1].plot(generations, avg_fitness, label=label, color=color, linestyle=linestyle, linewidth=1.5)

            if not num_species.empty:
                self.axes[2].plot(generations, num_species, label=label, color=color, linestyle=linestyle, linewidth=1.5)
        
        if max_generations_overall > 0:
            for ax in self.axes:
                ax.set_xlim(left=0, right=max_generations_overall * 1.05) 
        self.axes[0].set_title("Максимальний фітнес за поколіннями", fontsize=10)
        self.axes[0].set_ylabel("Макс. фітнес", fontsize=9)
        self.axes[0].legend(fontsize='small', loc='best')

        self.axes[1].set_title("Середній фітнес за поколіннями", fontsize=10)
        self.axes[1].set_ylabel("Сер. фітнес", fontsize=9)
        self.axes[1].legend(fontsize='small', loc='best')

        self.axes[2].set_title("Кількість видів за поколіннями", fontsize=10)
        self.axes[2].set_ylabel("К-ть видів", fontsize=9)
        self.axes[2].set_xlabel("Покоління", fontsize=9)
        self.axes[2].legend(fontsize='small', loc='best')
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiRunComparerApp(root)
    root.mainloop()