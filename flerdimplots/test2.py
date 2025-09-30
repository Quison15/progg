import numpy as np
import matplotlib.pyplot as plt

# Parametrar för vågen
A = 1            # amplitud
λ = 2            # våglängd (meter)
T = 4            # period (sekunder)
k = 2 * np.pi / λ  # vågtal
ω = 2 * np.pi / T  # vinkelhastighet
v = λ / T         # vågens hastighet

# Vågfunktion: vänstergående våg
def wave(x, t):
    return A * np.cos(k * x + ω * t)  # + => vänster, - => höger

# Tidpunkt t0 och plats x0
t0 = 2.0
x0 = 1.0

# Skapa grafer
x_vals = np.linspace(0, 4, 1000)
t_vals = np.linspace(0, 4, 1000)

y_space = wave(x_vals, t0)
y_time = wave(x0, t_vals)

# Rita vågens profil i rummet vid tid t0
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_space)
plt.title(f"Vågen som funktion av x vid t = {t0:.1f} s")
plt.xlabel("x (m)")
plt.ylabel("y(x, t₀)")
plt.grid(True)

# Rita vågens utveckling i tiden vid plats x0
plt.subplot(1, 2, 2)
plt.plot(t_vals, y_time)
plt.title(f"Vågen som funktion av t vid x = {x0:.1f} m")
plt.xlabel("t (s)")
plt.ylabel("y(x₀, t)")
plt.grid(True)

plt.tight_layout()
plt.show()
