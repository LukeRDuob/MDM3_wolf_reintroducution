# WeeklyApp.py

import solara
import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from WeeklyModel_ import WeeklySpeciesModel
#import WeeklyModel_veg


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



def make_vegetation_component():
    @solara.component
    def vegetation_plot(model):
        # Force rerender when the model advances
        current_step = model.steps

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(
            model.veg_value.T,
            origin="lower",
            cmap="YlGn",
            vmin=0,
            vmax=model.max_veg,
            extent=[0, model.width, 0, model.height],
        )

        ax.set_title(f"Vegetation state (step {current_step})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.colorbar(im, ax=ax, label="Vegetation")

        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    return vegetation_plot



model = WeeklySpeciesModel(
    max_steps=50000
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
        #make_vegetation_component(),

    ],
    name="Simplified Weekly Wolf-Deer Model",
)

Page