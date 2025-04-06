import numpy as np
import matplotlib.pyplot as plt

def deviation_angle(theta_inc_deg, alpha_deg=60, n=1.57):
    """
    Beräknar avlänkningsvinkeln (delta) i grader för ett prisma med apexvinkel alpha.
    
    Parametrar:
      theta_inc_deg: Infallande vinkel i grader (kan vara en array).
      alpha_deg: Apexvinkeln hos prismat i grader (standard 60°).
      n: Refraktionsindex för glaset (standard 1.57).
    
    Returnerar:
      delta_deg: Avlänkningsvinkeln i grader.
      overall_valid: Boolean-array som markerar de theta där lösningen är giltig (inga totala interna reflektioner).
    """
    # Omvandla grader till radianer
    theta_inc = np.radians(theta_inc_deg)
    alpha = np.radians(alpha_deg)
    
    # Första brytningen: luft (n=1) -> glas (n)
    # Snells lag: sin(theta_inc) = n * sin(r1)
    sin_r1 = np.sin(theta_inc) / n
    valid = sin_r1 <= 1.0  # måste vara giltigt för arcsin
    r1 = np.empty_like(theta_inc)
    r1[~valid] = np.nan
    r1[valid] = np.arcsin(sin_r1[valid])
    
    # Andra ytan: inuti prismat träffar ljuset med vinkel r2 mot normalen
    r2 = alpha - r1  # r2 i radianer
    
    # Vid utträdet: Snells lag: n*sin(r2) = sin(theta2)
    sin_theta2 = n * np.sin(r2)
    valid2 = sin_theta2 <= 1.0  # Undvik totala interna reflektioner
    overall_valid = valid & valid2
    theta2 = np.empty_like(theta_inc)
    theta2[~overall_valid] = np.nan
    theta2[overall_valid] = np.arcsin(sin_theta2[overall_valid])
    
    # Avlänkningsvinkel: delta = theta_inc + theta2 - alpha
    delta = np.where(overall_valid, theta_inc + theta2 - alpha, np.nan)
    delta_deg = np.degrees(delta)
    
    return delta_deg, overall_valid

# Del (a): Använd konstant brytningsindex n = 1.57
theta_range = np.linspace(30, 90, 1000)  # Infallande vinklar i grader
delta_deg, valid = deviation_angle(theta_range, alpha_deg=60, n=1.57)

plt.figure()
plt.plot(theta_range[valid], delta_deg[valid])
plt.xlabel("Infallande vinkel (θ, grader)")
plt.ylabel("Avlänkningsvinkel (δ, grader)")
plt.title("Avlänkningsvinkel vs. Infallande vinkel (n=1.57)")
plt.grid(True)
plt.show()


def refractive_index(lambda_nm, A1=1.522, A2=4590):
    """
    Beräknar refraktionsindexet n för en given våglängd (i nm) med materialkonstanterna A1 och A2.
    """
    return A1 + A2/(lambda_nm**2)

wavelengths = [400, 550, 700]  # Våglängder i nm
colors = ['blue', 'green', 'red']  # Färgkodning för respektive våglängd

plt.figure()
for wl, col in zip(wavelengths, colors):
    n_wl = refractive_index(wl)
    delta_wl, valid = deviation_angle(theta_range, alpha_deg=60, n=n_wl)
    plt.plot(theta_range[valid], delta_wl[valid], label=f"{wl} nm, n={n_wl:.3f}", color=col)

plt.xlabel("Infallande vinkel (θ, grader)")
plt.ylabel("Avlänkningsvinkel (δ, grader)")
plt.title("Avlänkningsvinkel vs. Infallande vinkel för olika våglängder")
plt.legend()
plt.grid(True)
plt.show()

