'''
From the coupled system of equations:
dD / dt = r_D · D (1 − D / K_D) − aWD
dW / dt = baWD - mW
This script:
(1) Builds a function that passes the args to ODE solver
(2) Numerically solves for equilibria of the system
(3) Solve for parameter ranges that give stability
(4) Find specific parameter values that produce limit cycles
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import integrate
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')



# # Known Paramaters 
# cmina = 1.138      # yearly deer growth rate (birthrate - natural death rate)
# d   = 0.125     # wolf mortality rate (av life expectance of 8 years)
D0 = 4285    # deer population of affric and kintail 2021

# # Unkown Paramaters to sweep
# e  = 17/7000 # 0.005   # wolf predation rate on deer -> 17 deer killed per wolf per year 
# beta =    0.004 # 0.0002917152   # conversion efficiency of deer to wolf births
# W0_values = np.linspace(1, 50, 10).astype(int)
W0 = 5


D_star = 7000  # yellowstone deer equilibrium estimate 
cmina = 1.138      # deer birth rate
e = 17/D_star  # attack rate
d = 1/14       # wolf death rate
beta = d/(D_star*e)   # conversion efficiency
h = 0.03       # handling time
K = 15000      # deer carrying capacity



param_arr = [cmina, d, e, beta]
t_eval = np.arange(0, 100, 1) # 50 yearly timesteps 
y0 = [D0,W0]




def lot_volt(t, ic_arr, param_arr):
    D, W = ic_arr
    cmina, d, e, beta = param_arr

    dDdt = cmina * D - e*W*D
    dWdt = beta*W*D - d*W
    return [dDdt, dWdt]


def solve_system(param_arr, ic_arr, t_eval):
    sol = solve_ivp(lot_volt, (t_eval[0], t_eval[-1]), ic_arr, t_eval=t_eval, args=(param_arr,))
    return sol


def plot_time_series(sol):
    # plot time series
    plt.figure(figsize=(10,6))
    plt.plot(sol.t, sol.y[0], label="Deer")
    plt.plot(sol.t, sol.y[1], label="Wolves")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Predator-Prey System")
    plt.show()

def plot_phase(sol):
    # plot phase plot
    plt.figure(figsize=(10,6))
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("Deer")
    plt.ylabel("Wolves")
    plt.title("Phase Plot")
    plt.show()

def plot_w0_arr(W0_values):
    # Plotting array of centers for W0 
    plt.figure()
    for w0 in W0_values:
        y0 = [D0, w0]
        sol = solve_ivp(lot_volt, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, args=(param_arr,))
        plt.plot(sol.y[0], sol.y[1], "-", label = f"Wolf IC= {y0[1]}") 
    plt.xlabel("Deer")
    plt.ylabel("Wolves")
    plt.legend()
    plt.title("Deer vs Wolves")
    plt.show()

# def param_stab


def animate_sol(ic_arr = y0, param_arr = param_arr, t_eval = t_eval):

    # Solve system once
    sol = solve_system(param_arr, ic_arr, t_eval)
    # Create figure with two subplots 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Visulatisation code
    # Set labels and limits for both subplots
    ax1.set_xlabel("Time", fontsize=12); ax1.set_ylabel("Population", fontsize=12)
    ax2.set_xlabel("Deer", fontsize=12); ax2.set_ylabel("Wolves", fontsize=12)
    # Set fixed limits from max sol vals
    ax1.set_xlim(sol.t[0], sol.t[-1])
    ax1.set_ylim(0, sol.y.max() * 1.1)  
    ax2.set_xlim(0, sol.y[0].max() * 1.1)
    ax2.set_ylim(0, sol.y[1].max() * 1.1)
    # Remove top and right spines 
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    # Set titles
    ax1.set_title("Time Series"); ax2.set_title("Phase Plot")
    fig.suptitle(f"Lotka-Volterra Wolf-Deer Dynamics Introducing n={ic_arr[1]} Wolves", y=0.98, fontsize = 16)

    # Create empty line objects
    line_deer, = ax1.plot([], [], label="Deer")
    line_wolf, = ax1.plot([], [], label="Wolves")
    line_phase, = ax2.plot([], [], color = 'red')
    ax1.legend()

    # Create text object for population 
    pop_text = fig.text(0.5, 0.84, '',
                    ha='center', fontsize=11, 
                    fontfamily='monospace', fontweight='bold')

    # Create time object 
    time_text = fig.text(0.5, 0.88, '',
                     ha='center', fontsize=14, color='gray')



    # Function returning single frame 
    # set_data mutates the original line object 
    def func(frame):
        line_deer.set_data(sol.t[:frame], sol.y[0][:frame])
        line_wolf.set_data(sol.t[:frame], sol.y[1][:frame])
        line_phase.set_data(sol.y[0][:frame], sol.y[1][:frame])
        time_text.set_text(f'Year {int(sol.t[frame]):02d}')
        pop_text.set_text(f'Deer: {int(sol.y[0][frame]):,}     Wolves: {int(sol.y[1][frame]):,}')
        return line_deer, line_wolf, line_phase, time_text, pop_text
    # Create animation 
    ani = FuncAnimation(fig, func, frames=len(sol.t), repeat=False)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    ani.save('lotka_volterra.gif', writer='pillow', fps=10)
    plt.show()

# animate_sol()

# plot_w0_arr(W0_values)

sol =  solve_system(param_arr, y0, t_eval)

plot_time_series(sol)