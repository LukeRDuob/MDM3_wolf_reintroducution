import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# dummy parameters (taken from chatgpt)

r_D = 0.5      # deer intrinsic growth rate
K_D = 500      # deer carrying capacity
a   = 0.01     # wolf predation rate on deer
b   = 0.1      # conversion efficiency (deer -> wolf births)
m   = 0.2      # wolf mortality rate

r_V = 0.8      # vegetation growth rate
c   = 0.005    # deer grazing rate on vegetation

r_C = 0.6      # crop growth rate
K_C = 300      # crop carrying capacity
h   = 0.004    # deer grazing rate on crops
p   = 0.002    # wolf damage to crops



# system of ODEs

def verhulst_lotka_volterra(t, y):
    D, W, V, C = y

    dDdt = r_D * D * (1 - D / K_D) - a * W * D
    dWdt = b * a * W * D - m * W
    dVdt = r_V * V * (1 - V / K_D) - c * D * V
    dCdt = r_C * C * (1 - C / K_C) - h * D * C - p * W * C

    return [dDdt, dWdt, dVdt, dCdt]


# init conds

D0 = 100   # init deer
W0 = 20    # init wolves
V0 = 400   # init vegetation
C0 = 200   # init crops

y0 = [D0, W0, V0, C0]



# time

t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)


# solve

solution = solve_ivp(verhulst_lotka_volterra, t_span, y0, t_eval=t_eval)


# plot

plt.figure(figsize=(10,6))
plt.plot(solution.t, solution.y[0], label="Deer")
plt.plot(solution.t, solution.y[1], label="Wolves")
plt.plot(solution.t, solution.y[2], label="Vegetation")
plt.plot(solution.t, solution.y[3], label="Crops")

plt.xlabel("Time")
plt.ylabel("Population / Biomass")
plt.legend()
plt.title("Verhulst–Lotka–Volterra System")
plt.grid()
plt.show()