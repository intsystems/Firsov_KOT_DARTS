import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

# 1) Отключаем стандартные оси (рамку и сетку)
ax._axis3don = False  # убирает привычный бокс и «штатные» оси

# 2) Определяем разумные пределы по каждой оси,
#    чтобы ноль попал "внутрь" видимой области
ax.set_xlim(-0.2, 1.0)
ax.set_ylim(-0.2, 0.8)
ax.set_zlim(-0.2, 0.6)

# 3) Рисуем 3 оси через (0,0,0)
#    Для наглядности возьмём их длину побольше
#    (в данном случае рисуем от -0.2 до 1.0 по x и т.д.)
ax.plot([0, 0.8], [0, 0], [0, 0], color='black', lw=2)
ax.plot([0, 0], [0, 0.8], [0, 0], color='black', lw=2)
ax.plot([0, 0], [0, 0], [0, 0.8], color='black', lw=2)

# 4) Подпишем оси текстом на концах
ax.text(1, 0, 0, 'pool', color='r', fontsize=12)
ax.text(0, 0.8, 0, 'conv', color='g', fontsize=12)
ax.text(0, 0, 0.8, 'identity', color='b', fontsize=12)

# 5) Теперь добавим ваш «треугольник» (плоскость через 3 точки)
p1 = np.array([0,    0,    15/28])
p2 = np.array([0,    13/28, 0   ])
p3 = np.array([0.001,  0,    0   ])
points = np.vstack((p1, p2, p3))

tri = Poly3DCollection([points], alpha=0.4, facecolor='grey', edgecolor='black')
ax.add_collection3d(tri)

# Также отобразим сами точки для наглядности
ax.scatter(p1[0],p1[1],p1[2],s=40, color='blue')
ax.scatter(p2[0],p2[1],p2[2], s=40, color='green')
ax.scatter(p3[0],p3[1],p3[2], s=40, color='red')

ax.set_box_aspect((1,1,1))
ax.view_init(elev=30, azim=30)
plt.show()
