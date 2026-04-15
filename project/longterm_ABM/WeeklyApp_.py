# WeeklyApp.py

import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from WeeklyModel_ import WeeklySpeciesModel


AGENT_COLOURS = {
    "DeerHerd": "orange",
    "WolfPack": "blue", 
}


def apply_colours(ax):
    for line in ax.get_lines():
        label = line.get_label()

        if label == "Total Deer":
            line.set_color("orange")
        elif label == "Total Wolves":
            line.set_color("blue")
        elif label == "Weekly Deer Killed":
            line.set_color("red")
        elif label == "Pack 1 Size":
            line.set_color("navy")
        elif label == "Pack 2 Size":
            line.set_color("royalblue")
        elif label == "Pack 3 Size":
            line.set_color("slateblue")
        elif label == "Pack 4 Size":
            line.set_color("mediumpurple")
        elif label == "Pack 5 Size":
            line.set_color("darkviolet")

    ax.legend()


def agent_draw(agent):
    if agent.species == "DeerHerd":
        if agent.group_size <= 0:
            return {"color": "none", "size": 0}
        return {
            "color": AGENT_COLOURS["DeerHerd"],
            "size": max(10, agent.group_size * 0.3),
        }

    elif agent.species == "WolfPack":
        if agent.pack_size <= 0:
            return {"color": "none", "size": 0}
        return {
            "color": AGENT_COLOURS["WolfPack"],
            "size": max(15, agent.pack_size * 3),
        }


model = WeeklySpeciesModel(
    max_steps=150000
)


Page = SolaraViz(
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
            ["Total Deer"],
            post_process=apply_colours,
        ),
        make_plot_component(
            ["Total Deer Killed"],
            post_process=apply_colours,
        ),

    ],
    name="Simplified Weekly Wolf-Deer Model",
)

Page