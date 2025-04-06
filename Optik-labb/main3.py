import numpy as np
import matplotlib.pyplot as plt

# Parametrar
f = 3              # Fokallängd (avstånd från ljuskälla till lins)
lens_x = 0         # Linsens position på x-axeln

# Ljuskällans position
source = (-f, 0)

# Vinklar (i grader) för de divergerande strålarna
angles_deg = [-15, 0, 15]
angles = np.radians(angles_deg)

# Skapa figur
plt.figure(figsize=(8,6))

# Rita linsen som en vertikal linje vid x=0
plt.plot([lens_x, lens_x], [-4, 4], 'k-', lw=2, label='Objektiv (lins)')

# Rita ljuskällan som en punkt
plt.plot(source[0], source[1], 'ro', label='Divergent ljuskälla (vid f)')

# Rita strålarna:
for phi in angles:
    # Stråle från ljuskällan till linsen
    x1 = np.linspace(source[0], lens_x, 100)
    y1 = (x1 - source[0]) * np.tan(phi) + source[1]
    plt.plot(x1, y1, 'b-')
    
    # Vid linsen ändras riktningen så att strålarna blir horisontella (kollimerade)
    y_lens = (lens_x - source[0]) * np.tan(phi) + source[1]
    x2 = np.linspace(lens_x, 4, 100)
    y2 = np.full_like(x2, y_lens)
    plt.plot(x2, y2, 'b-')
    
    # Markera skärningspunkten med linsen
    plt.plot(lens_x, y_lens, 'ko')

# Markera fokallängden med en dubbelriktad pil
plt.annotate("", xy=(lens_x, -0.5), xytext=(source[0], -0.5), 
             arrowprops=dict(arrowstyle="<->", color='green', lw=2))
plt.text((lens_x + source[0]) / 2 - 0.5, -0.8, 'f', color='green', fontsize=12)

# Rita optiska axeln
plt.axhline(0, color='gray', linestyle='--')

# Inställningar för diagrammet
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Strålkonstruktion: Kollimering av divergent ljuskälla')
plt.legend()
plt.grid(True)
plt.show()
