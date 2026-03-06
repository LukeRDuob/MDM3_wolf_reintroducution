import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# parameters

# deer
r_D = 0.35
K_D = 4000
m_D = 0.08

# wolves
wolf_kill = 20          # deer per wolf per year
H = 400                 # half saturation
wolf_birth = 0.3
wolf_mort = 0.12
wolf_delay = 0.3        # years

# vegetation
r_V = 1.2
K_V = 10000
grazing = 0.00015

# crops (very weak)
r_C = 0.4
K_C = 500
crop_grazing = 0.0003

# livestock (weak vegetation competition)
r_L = 0.25
K_L = 800
veg_comp = 0.0005


# the cull bit
cull_threshold = 3000
cull_rate = 0.15        # 15% per year


# system

history_t = []
history_W = []


def wolf_delayed(t):

    if len(history_t) < 2:
        return history_W[-1]

    delay_time = t - wolf_delay

    if delay_time <= history_t[0]:
        return history_W[0]

    return np.interp(delay_time, history_t, history_W)


def ecosystem(t, y):

    D, W, V, C, L = y

    history_t.append(t)
    history_W.append(W)

    # vegetation dependence
    veg_factor = V / (V + 200)

    # predator saturation
    predation = wolf_kill * W * D / (D + H)

    # delayed wolves
    W_delay = wolf_delayed(t)

    food_factor = D / (D + H)

    wolf_births = wolf_birth * W_delay * food_factor

    # seasonal cull
    cull = 0
    year_frac = t % 1

    if D > cull_threshold and 0.75 < year_frac < 1.0:
        cull = (cull_rate / 0.25) * D

    # equations

    dDdt = r_D * D * (1 - D / K_D) * veg_factor - predation - m_D * D- cull
    dWdt = wolf_births - wolf_mort * W
    dVdt = r_V * V * (1 - V / K_V) - grazing * D * V - veg_comp * L * V
    dCdt = r_C * C * (1 - C / K_C) - crop_grazing * D * C
    dLdt = r_L * L * (1 - L / K_L)

    return [dDdt, dWdt, dVdt, dCdt, dLdt]


# init conds

D0 = 800
W0 = 20
V0 = 6000
C0 = 200
L0 = 300

y0 = [D0, W0, V0, C0, L0]


# time

t_span = (0, 50)
t_eval = np.linspace(0, 50, 4000)


# solve

solution = solve_ivp(ecosystem, t_span, y0, t_eval=t_eval)


# plot

plt.figure(figsize=(10,6))

plt.plot(solution.t, solution.y[0], label='Deer')
plt.plot(solution.t, solution.y[1], label='Wolves')
plt.plot(solution.t, solution.y[2], label='Vegetation')
plt.plot(solution.t, solution.y[3], label='Crops')
plt.plot(solution.t, solution.y[4], label='Livestock')

plt.yscale('log')

plt.xlabel('Years')
plt.ylabel('Population / Biomass')
plt.title('Scottish Deer–Wolf Model Mix')
plt.legend()
plt.grid()

plt.show()