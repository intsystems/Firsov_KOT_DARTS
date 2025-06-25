import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def plot_ops_radar(ops: Dict[str, int],
                   title: str = "Роза ветров операций",
                   face_color: str = "indianred",
                   edge_color: str = "firebrick",
                   grid_color: str = "mediumpurple",
                   fill_alpha: float = 0.35) -> None:
    """
    Строит «спайдер»-диаграмму (радари) по операциям модели.

    Parameters
    ----------
    ops : Dict[str, int]
        Ключи – названия операций (3-7 штук), значения – их количество.
    title : str
        Заголовок графика.
    face_color : str
        Цвет заливки многоугольника (RGBA / HTML / X11).
    edge_color : str
        Цвет линии многоугольника и точек.
    grid_color : str
        Цвет пунктирных концентрических «колец».
    fill_alpha : float
        Прозрачность заливки (0–1).
    """

    # --- подготовка данных --------------------------------------------------
    names, counts = zip(*ops.items())
    N = len(names)
    max_count = max(counts)

    # Углы для осей (равномерно по окружности) + дублируем первый,
    # чтобы замкнуть многоугольник
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values = list(counts) + [counts[0]]  # замыкаем

    # --- создаём оси ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")   # «север» вертикально вверх
    ax.set_theta_direction(-1)        # обход по часовой стрелке
    ax.set_ylim(0, max_count)

    # --- концентрические пунктирные кольца ----------------------------------
    # шаг кольца = 1, можно изменить при желании
    for r in range(1, max_count + 1, 3):
        ax.plot(np.linspace(0, 2 * np.pi, 360),
                [r] * 360,
                linestyle=":",
                color=grid_color,
                linewidth=1)

        # подпись уровня (расположена сверху)
        ax.text(np.pi / 2, r, f"{r}", va="center", ha="center",
                fontsize=19, color=grid_color, alpha=.9)

    # --- рисуем многоугольник ----------------------------------------------
    ax.plot(angles, values, color=edge_color, linewidth=2)
    ax.fill(angles, values, color=face_color, alpha=fill_alpha)

    # точки на вершинах
    ax.scatter(angles[:-1], values[:-1], color=edge_color, s=40, zorder=3)

    # --- подписи осей --------------------------------------------------------
    for angle, name in zip(angles[:-1], names):
        # angle_deg = np.degrees(angle)
        # выдвигаем подпись чуть наружу последнего кольца
        ax.text(angle,
                max_count + max_count * 0.08,
                name,
                ha="center",
                va="center",
                fontsize=24,
                # rotation=angle_deg,
                rotation_mode="anchor")

    # --- оформление ----------------------------------------------------------
    ax.set_title(title, pad=30, fontsize=14, weight="bold")
    ax.set_xticks([])           # убираем стандартные метки углов
    ax.set_yticks([])           # убираем стандартные радиальные деления
    ax.grid(False)              # сетку не показываем (мы её нарисовали сами)

    plt.tight_layout()
    plt.show()


# -------------------- пример использования ----------------------------------
if __name__ == "__main__":
    ops = {
        "pooling": 18,
        "convolution": 9,
        "identity": 1,
        # "conv_5_5": 4,
        # "max_pool": 7
    }
    plot_ops_radar(ops, title="")
