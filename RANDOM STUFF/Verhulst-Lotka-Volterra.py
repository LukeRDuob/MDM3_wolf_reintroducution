import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# dummy parameters (initial ones taken from chatgpt)

r_D = 0.1      # deer intrinsic growth rate
K_D = 500      # deer carrying capacity
a   = 0.05     # wolf predation rate on deer
b   = 0.04      # conversion efficiency (deer -> wolf births)
m   = 0.2      # wolf mortality rate

r_V = 0.8      # vegetation growth rate
c   = 0.001    # deer grazing rate on vegetation
g   = 0.002

r_C = 0.6      # crop growth rate
K_C = 300      # crop carrying capacity
h   = 0.004    # deer grazing rate on crops
p   = 0.002    # wolf damage to crops

r_L = 0.5      # livestock growth rate
K_L = 1000     # livestock carrying capacity
s   = 0.002



# system of ODEs

def verhulst_lotka_volterra(t, y):
    D, W, V, C = y

    dDdt = r_D * D * (1 - D / K_D) - a * W * D
    dWdt = b * a * W * D - m * W
    dVdt = r_V * V * (1 - V / K_D) - c * D * V
    dCdt = r_C * C * (1 - C / K_C) - h * D * C - p * W * C

    return [dDdt, dWdt, dVdt, dCdt]

def vlv_livestock(t, y):
    D, W, V, C, L = y

    dDdt = r_D * D * (1 - D / K_D) - a * W * D
    dWdt = b * a * W * D - m * W
    dVdt = r_V * V * (1 - V / K_D) - c * D * V - g * L * V
    dCdt = r_C * C * (1 - C / K_C) - h * D * C - p * W * C
    dLdt = r_L * L * (1 - L / K_L) - h * D * L - s * D * L

    return [dDdt, dWdt, dVdt, dCdt, dLdt]

# init conds

D0 = 100   # init deer
W0 = 20    # init wolves
V0 = 400   # init vegetation
C0 = 200   # init crops
L0 = 300   # init livestock

y0 = [D0, W0, V0, C0]
y0_l = [D0, W0, V0, C0, L0]



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



# solve

solution = solve_ivp(vlv_livestock, t_span, y0_l, t_eval=t_eval)


# plot

plt.figure(figsize=(10,6))
plt.plot(solution.t, solution.y[0], label="Deer")
plt.plot(solution.t, solution.y[1], label="Wolves")
plt.plot(solution.t, solution.y[2], label="Vegetation")
plt.plot(solution.t, solution.y[3], label="Crops")
plt.plot(solution.t, solution.y[4], label="Livestock")

plt.xlabel("Time")
plt.ylabel("Population / Biomass")
plt.legend()
plt.title("Verhulst–Lotka–Volterra System with Livestock")
plt.grid()
plt.show()