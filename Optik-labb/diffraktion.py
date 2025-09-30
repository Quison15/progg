import cv2
import numpy as np
import matplotlib.pyplot as plt

# Läs in bilden i gråskala
img = cv2.imread('fresnel.jpg', cv2.IMREAD_GRAYSCALE)

# Välj en horisontell rad (t.ex. mitt i bilden)
row = img.shape[0] // 2
intensity_profile = img[row, :]  # tar pixelvärden längs en rad

# Normalisera till maxintensitet (typiskt långt från kanten)
normalized = intensity_profile / np.max(intensity_profile)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(normalized, label='Normaliserad intensitet')
plt.xlabel('Pixlar längs skärmen')
plt.ylabel('Relativ intensitet')
plt.title('Intensitetsprofil från Fresneldiffraktion')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
