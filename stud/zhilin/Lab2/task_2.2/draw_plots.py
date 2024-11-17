import numpy as np
import matplotlib.pyplot as plt


figure, axes = plt.subplots(1)

angle = np.linspace(0, 2 * np.pi, 200)
x1 = 4 * np.cos(angle)
x2 = 4 * np.sin(angle)
axes.plot(x1, x2)

x2 = np.linspace(-5, 2.5, 500)
x1 = list(map(lambda i: np.e**i - 4, x2))
axes.plot(x1, x2)


plt.xticks(np.arange(min(*x1, *x2)-1, max(*x1, *x2)+1, 1.0))
plt.yticks(np.arange(min(*x1, *x2)-1, max(*x1, *x2)+1, 1.0))
axes.set_aspect(1)
plt.grid()
plt.show()
