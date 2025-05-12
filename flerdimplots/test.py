import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Skapa ett r-nät för xy-planet
r = np.linspace(0, 1, 100)
theta = np.linspace(0, 2*np.pi, 100)
r, theta = np.meshgrid(r, theta)

# Konvertera till x, y
x = r * np.cos(theta)
y = r * np.sin(theta)

# Undre gräns (planet z = 1)
z_bottom = np.ones_like(x)

# Övre gräns (sfärens yta)
z_top = np.sqrt(4 - x**2 - y**2)

# Skapa 3D-plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Rita övre kupolen
ax.plot_surface(x, y, z_top, alpha=0.7, color='lightblue', edgecolor='k', linewidth=0.2, label='Övre sfär')

# Rita undre planet
ax.plot_surface(x, y, z_bottom, alpha=0.5, color='lightcoral', edgecolor='k', linewidth=0.2)

# Ändra vy och etiketter
ax.view_init(elev=30, azim=45)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Kroppen K: mellan z = 1 och z = √(4 - x² - y²) över x² + y² ≤ 1')

plt.tight_layout()
plt.show()
