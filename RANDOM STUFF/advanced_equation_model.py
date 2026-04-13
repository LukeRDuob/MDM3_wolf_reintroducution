import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# -----------------------
# PARAMETERS
# ----------------------

# deer
r_D = 0.35 / 12
K_D = 40000 # was 4000
m_D = 0.03 / 12

# wolves
wolf_kill = 18 / 12      # ~ realistic upper bound
H = 8000  # was 300             # prey half-saturation
alpha = 0.15          # wolf interference (NEW)
wolf_birth = 0.28 / 12
wolf_mort = 0.15 / 12
wolf_delay = 0.5 * 12   # 6 months   # more realistic than 1 year

# minimum viable wolf pack — roughly 5 wolves can persist
# on small deer numbers via scavenging, small prey, etc.
WOLF_MIN = 5


# vegetation
r_V = 1.1 / 12
K_V = 90000 # was 9000
grazing = 0.000012 / 12 # was 0.00012 / 12

# crops (very weak)
r_C = 0.3 / 12
K_C = 4000 # was 400
crop_grazing = 0.000005 / 12 # was 0.00005 / 12

# livestock (weak competition)
r_L = 0.2 / 12
K_L = 7000 # was 700
veg_comp = 0.000008 / 12 # was 0.00008 / 12

# cull
cull_threshold = 22000 # was 2200
cull_rate = 0.15

# -----------------------
# INITIAL CONDITIONS
# ----------------------

# have all been multiplied bby 10
D0 = 18000
W0 = 70
V0 = 60000
C0 = 2000
L0 = 3000

y0 = [D0, W0, V0, C0, L0]
s_y0 = [D0, W0, V0]


# TIME

# t_span = (0, 50)
# t_eval = np.linspace(0, 50, 4000)
t_span = (0, 40 * 12)  # years to months
t_eval = np.linspace(0, 40 * 12, 4000)


# -----------------------
# DELAY BUFFER (bounded)
# ----------------------

history_t = []
history_W = []

MAX_HISTORY = 5000  # prevents memory blow-up


def wolf_delayed(t):

    if len(history_t) < 5:
        return history_W[-1]

    delay_time = t - wolf_delay

    if delay_time <= history_t[0]:
        return history_W[0]

    return np.interp(delay_time, history_t, history_W)


# -----------------------
# SYSTEM
# ----------------------

def ecosystem(t, y):

    D, W, V, C, L = y

    # store history (bounded)
    history_t.append(t)
    history_W.append(W)

    if len(history_t) > MAX_HISTORY:
        history_t.pop(0)
        history_W.pop(0)

    # -----------------------
    # ecological modifiers
    # ----------------------

    # vegetation dependence (weak but prevents collapse survival)
    veg_factor = V / (V + 300)

    # winter mortality ---- new bit ------------
    # winter = 0.02 * (1 + np.cos(2 * np.pi * t))  # peaks in winter
    winter = 0.005 * (1 + np.cos(2 * np.pi * t / 12))

    # predator saturation + interference (NEW)
    predation = wolf_kill * W * D / (D + H + alpha * W + 200)

    # # delayed wolves
    # W_delay = wolf_delayed(t)
    #
    # food_factor = D / (D + H + 1e-9)
    #
    # wolf_births = wolf_birth * W_delay * food_factor

    ########## WOLVES #########
    # FOOD AVAILABILITY
    food_factor = D / (D + H + 1e-9)
    W_delay = wolf_delayed(t)

    # STARVATION DYNAMICS (FAST COLLAPSE)
    critical_food = 0.3

    # starvation level (0 to 1)
    starvation = max(0, (critical_food - food_factor) / critical_food)
    # makes it VERY sharp
    starvation = starvation ** 2

    # BIRTHS (collapse)
    wolf_births = wolf_birth * W_delay * food_factor * (1 - starvation)

    # FAST STARVATION DEATH
    # baseline mortality
    base_death = wolf_mort * W

    # starvation death (dominates when food is low)
    starvation_death = 2.5 * starvation * W  # <-- THIS is the key term

    # recovery term — small positive nudge when W < WOLF_MIN
    # models immigration / scavenging survival of minimal pack
    # --- only active when wolves are critically low AND some deer remain.
    rescue = 0.0
    if W < WOLF_MIN and D > 500:
        rescue = 0.05 * (WOLF_MIN - W)  # gently pulls W back to ~WOLF_MIN

    dWdt = wolf_births - base_death - starvation_death

    #####################################

    # -----------------------
    # smooth seasonal cull --- new bit
    # ----------------------

    # year_frac = t % 1
    year_frac = (t % 12) / 12

    cull_window = np.exp(-((year_frac - 0.875) ** 2) / 0.002)

    cull = 0
    # if D > cull_threshold:
    #     cull = cull_rate * cull_window * D
    #     # cull = (cull_rate / np.sqrt(2 * np.pi * 0.002)) * cull_window * D
    month = t % 12

    if D > cull_threshold and 9 <= month <= 11:
        cull = (cull_rate / 3) * D

    # -----------------------
    # equations
    # ----------------------

    dDdt = r_D * D * (1 - D / K_D) * veg_factor - predation - (m_D + winter) * D - cull

    # dWdt = wolf_births - wolf_mort * W

    dVdt = r_V * V * (1 - V / K_V) - grazing * D * V - veg_comp * L * V

    dCdt = r_C * C * (1 - C / K_C) - crop_grazing * D * C

    dLdt = r_L * L * (1 - L / K_L)

    return [dDdt, dWdt, dVdt, dCdt, dLdt]


# -----------------------
# SLIM MODEL
# ----------------------

def slim_ecosystem(t, y):

    D, W, V = y

    # -----------------------
    # DELAY BUFFER
    # -----------------------

    history_t.append(t)
    history_W.append(W)

    if len(history_t) > MAX_HISTORY:
        history_t.pop(0)
        history_W.pop(0)

    # -----------------------
    # ECOLOGICAL MODIFIERS
    # -----------------------

    veg_factor = V / (V + 300)

    # match full model winter
    winter = 0.005 * (1 + np.cos(2 * np.pi * t))

    # FIXED predation (same as full model)
    predation = wolf_kill * W * D / (D + H + alpha * W + 200)

    # -----------------------
    # WOLF DYNAMICS (MATCH FULL MODEL)
    # -----------------------

    food_factor = D / (D + H + 1e-9)
    W_delay = wolf_delayed(t)

    critical_food = 0.3

    starvation = max(0, (critical_food - food_factor) / critical_food)
    starvation = starvation ** 2  # less sharp than cubic collapse

    # births collapse under starvation
    wolf_births = wolf_birth * W_delay * food_factor * (1 - starvation)

    # mortality
    base_death = wolf_mort * W
    starvation_death = 2.5 * starvation * W # the 2.5 was 3.5

    dWdt = wolf_births - base_death - starvation_death

    # -----------------------
    # CULL (same as full model)
    # -----------------------

    year_frac = t % 1
    cull_window = np.exp(-((year_frac - 0.875) ** 2) / 0.002)

    cull = 0
    if D > cull_threshold:
        cull = cull_rate * cull_window * D

    # -----------------------
    # EQUATIONS
    # -----------------------

    dDdt = (
        r_D * D * (1 - D / K_D) * veg_factor
        - predation
        - (m_D + winter) * D
        - cull
    )

    dVdt = r_V * V * (1 - V / K_V) - grazing * D * V

    return [dDdt, dWdt, dVdt]


# -----------------------
# EXTINCTION EVENT
# ----------------------

def extinction_event(t, y):
    return min(y[0] - 1, y[1] - 1)

extinction_event.terminal = True
extinction_event.direction = -1





# --- FULL MODEL
history_t.clear()
history_W.clear()

full_solution = solve_ivp(
    ecosystem, t_span, y0,
    t_eval=t_eval,
    events=extinction_event,
    max_step=0.5
)

# --- SLIM MODEL
history_t.clear()
history_W.clear()

slim_solution = solve_ivp(
    slim_ecosystem, t_span, s_y0,
    t_eval=t_eval,
    events=extinction_event,
    max_step=0.5
)


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

labels_full = ["Deer","Wolves","Vegetation","Crops","Livestock"]
labels_slim = ["Deer","Wolves","Vegetation"]

colors = ["green", "red", "darkgreen"]  # deer, wolves, vegetation

# PLOTTING

# full and slim
for i, label in enumerate(labels_full):
    axes[0,0].plot(full_solution.t, full_solution.y[i],
                   label=f"{label} (full)")
for i, label in enumerate(labels_slim):
    axes[0,0].plot(slim_solution.t, slim_solution.y[i],
                   '--', label=f"{label} (slim)")

axes[0,0].set_title("Full vs Slim (Linear)")
axes[0,0].legend()
axes[0,0].grid()

# log version of full and slim
for i in range(len(labels_full)):
    axes[0,1].plot(full_solution.t, full_solution.y[i])
for i in range(len(labels_slim)):
    axes[0,1].plot(slim_solution.t, slim_solution.y[i], '--')

axes[0,1].set_title("Full vs Slim (Log)")
axes[0,1].set_yscale('log')
axes[0,1].grid()

# populations of wolves and deer

# wolf pop
axes[1,0].plot(full_solution.t, full_solution.y[1],
               color='red', label="Wolves (full)")
axes[1,0].plot(slim_solution.t, slim_solution.y[1],
               '--', color='red', label="Wolves (slim)")

axes[1,0].set_title("Wolf Population (Linear)")
axes[1,0].legend()
axes[1,0].grid()

# deer pop
axes[1,1].plot(full_solution.t, full_solution.y[0],
               color='green', label="Deer (full)")
axes[1,1].plot(slim_solution.t, slim_solution.y[0],
               '--', color='green', label="Deer (slim)")

axes[1,1].set_title("Deer Population (Linear)")
axes[1,1].legend()
axes[1,1].grid()

# general plot bits for everything

for ax in axes.flat:
    ax.set_xlabel("Months")

plt.suptitle("Full vs Slim Ecosystem Model Comparison", fontsize=14)
plt.tight_layout()
plt.show()