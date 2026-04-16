# WeeklyApp.py

import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from WeeklyModel import WeeklySpeciesModel
#from VegClass import Vegetation


AGENT_COLOURS = {
    "DeerHerd": "orange",
    "WolfPack": "blue",
    #"Vegetation": "#5e8354",
}


def apply_colours(ax):
    for line in ax.get_lines():
        if line.get_label() == "Total Deer":
            line.set_color("orange")
        elif line.get_label() == "Total Wolves":
            line.set_color("blue")
        elif line.get_label() == "Weekly Deer Killed":
            line.set_color("red")
        elif line.get_label() == "Total Saplings":
            line.set_color("#a2c399")
        elif line.get_label() == "Total Trees":
            line.set_color("#5e8354")
    ax.legend()


def agent_draw(agent):
    if agent.species == "DeerHerd":
        return {
            "color": AGENT_COLOURS["DeerHerd"],
            "size": max(10, agent.group_size * 2),
        }

    elif agent.species == "WolfPack":
        return {
            "color": AGENT_COLOURS["WolfPack"],
            "size": max(15, agent.pack_size * 4),
        }

    #elif agent.species == "Vegetation":
        return {
            "color": "none",
            "size": 0,
        }

'''
def draw_vegetation_overlay(ax, model):
    veg_agents = model.agents_by_type.get(Vegetation, [])

    if not veg_agents:
        return

    xs = [v.pos[0] for v in veg_agents]
    ys = [v.pos[1] for v in veg_agents]

    sapling_fractions = [
        v.saplings / v.max_saplings if v.max_saplings > 0 else 0
        for v in veg_agents
    ]

    sizes = [100 + 1200 * f for f in sapling_fractions]
    colours = [plt.cm.Greens(0.2 + 0.7 * f) for f in sapling_fractions]

    ax.scatter(xs, ys, s=sizes, c=colours, alpha=0.45, edgecolors="none")


def make_space_with_overlays(model):
    def space_with_overlays(ax):
        draw_vegetation_overlay(ax, model)
    return space_with_overlays
'''

model = WeeklySpeciesModel(
    max_steps=2080
)


page = SolaraViz(
    model,
    components=[
        make_space_component(
            agent_portrayal=agent_draw,
            backend="matplotlib",
            
        ),
        make_plot_component(
            ["Total Wolves"],
            post_process=apply_colours,
        ),

        make_plot_component(
            ["Weekly Deer Killed"],
            post_process=apply_colours,
        ),

        make_plot_component(
            ["Total Deer"],
            post_process=apply_colours,
        ),
        
        make_plot_component(
            ["Pack 1 Size", "Pack 2 Size", "Pack 3 Size", "Pack 4 Size", "Pack 5 Size",
              "Pack 6 Size", "Pack 7 Size"],
            post_process=apply_colours
        ),
        make_plot_component(
            ["Pack 1 Energy", "Pack 2 Energy", "Pack 3 Energy", "Pack 4 Energy",
              "Pack 5 Energy", "Pack 6 Energy", "Pack 7 Energy"],
            post_process=apply_colours
        ),
        #make_plot_component(
         #   ["Total Saplings", "Total Trees"],
        #   post_process=apply_colours,
        #),
    ],
    name="Weekly Wolf-Deer-Vegetation Model",
)

page