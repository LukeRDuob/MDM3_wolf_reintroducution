import solara
import matplotlib.pyplot as plt
from mesa.visualization import Slider, SolaraViz, make_space_component, make_plot_component
from Model import SpeciesModel
from VegetationClass import Vegetation
import pandas as pd
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

AGENT_COLOURS = {
    "Deer": "orange",
    "Wolf": "blue",
    "Lynx": "brown",
    "Sapling": "#a2c399",
    "Tree": "#5e8354",
}


def make_time_plot(metrics: list[str], post_process=None):
    def MakeTimePlot(model):
        return TimePlotMatplotlib(model, metrics, post_process=post_process)
    return (MakeTimePlot, 0)  # the tuple with page number is also required

@solara.component
def TimePlotMatplotlib(model, metrics, post_process=None):
    update_counter.get()  # <-- this is the key, hooks into Mesa's render cycle
    
    fig = Figure()
    ax = fig.subplots()
    
    df = model.datacollector.get_model_vars_dataframe()
   
    x = df.index * model.step_size  # or your step_size
    x_label = "Time (hours)"
    
    for metric in metrics:
        if metric in df.columns:
            ax.plot(x, df[metric], label=metric)
    
    ax.set_xlabel(x_label)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=False))
    ax.legend(loc="best")
    
    if post_process:
        post_process(ax)

    
    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")

def apply_colours(ax):
    """Reusable - applies AGENT_COLOURS to any plot"""
    for line in ax.get_lines():
        if line.get_label() in AGENT_COLOURS:
            line.set_color(AGENT_COLOURS[line.get_label()])
    ax.legend()

def agent_draw(agent):
    """Display the agents with assigned colours."""
    if agent.species == "Deer":
        return {"color": AGENT_COLOURS["Deer"], "size": 5}
    elif agent.species == "Wolf":
        return {"color": AGENT_COLOURS["Wolf"], "size": 5}
    elif agent.species == "Lynx":
        return {"color": AGENT_COLOURS["Lynx"], "size": 5} 
    elif agent.species == "Vegetation":
        return{"color":1}


import numpy as np
import solara
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mesa.visualization.utils import update_counter

from DeerClass import Deer
from LynxClass import Lynx
from WolfClass import Wolf


# Paths to your images
AGENT_IMAGES = {
    "Deer": "images/deer.png",
    "Wolf": "images/wolf.png",
    "Lynx": "images/lynx.png",
}

# Control image sizes
ZOOM_SIZE = {
    "Deer": 0.08,
    "Wolf": 0.08,
    "Lynx": 0.08,
}


def add_image(ax, image_path, x, y, zoom=0.1):
    img = np.array(Image.open(image_path))
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, xycoords="data")
    ax.add_artist(ab)


@solara.component
def animal_space(model):
    update_counter.get()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_aspect("equal")
    ax.set_title(f"{model.predator}–Deer Model")
    ax.axis("off")

    for agent in model.agents:
        x, y = agent.pos

        if isinstance(agent, Deer):
            add_image(ax, AGENT_IMAGES["Deer"], x, y, zoom=ZOOM_SIZE["Deer"])

        elif isinstance(agent, Wolf):
            add_image(ax, AGENT_IMAGES["Wolf"], x, y, zoom=ZOOM_SIZE["Wolf"])

        elif isinstance(agent, Lynx):
            add_image(ax, AGENT_IMAGES["Lynx"], x, y, zoom=ZOOM_SIZE["Lynx"])

    solara.FigureMatplotlib(fig)
    plt.close(fig)


def draw_vegetation_overlay(ax):
    """Draw vegetation patches based on sapling density only."""
    veg_agents = model.agents_by_type.get(Vegetation, [])

    if not veg_agents:
        return

    xs = [v.pos[0] for v in veg_agents]
    ys = [v.pos[1] for v in veg_agents]

    # Sapling fraction in each patch
    sapling_fractions = [
        v.saplings / v.max_saplings if v.max_saplings > 0 else 0
        for v in veg_agents
    ]

    # Blob size responds to sapling amount
    sizes = [
        100 + 2000 * f
        for f in sapling_fractions
    ]

    # Colour responds to sapling density
    # Using matplotlib's "Greens" colormap
    colours = [plt.cm.Greens(0.2 + 0.7 * f) for f in sapling_fractions]

    ax.scatter(xs, ys, s=sizes, c=colours, alpha=0.45, edgecolors="none")


def space_with_overlays (ax):
    # Draw elevation as background if avaliable
    if hasattr(model, "elevation_grid"):
        ax.imshow(
            model.elevation_grid,
            extent=[0, model.width, 0, model.height],
            origin="lower",
            cmap="terrain",
            alpha=0.6  # transparency so agents show
        )

    # Draw vegetation overlay
    draw_vegetation_overlay(ax)



# Initiate the model
model = SpeciesModel(
    max_steps=1000000, 
    use_base=False,
    use_veg=False,
    )


# Create space components
elevation_space_component = make_space_component(
    agent_portrayal=agent_draw,
    backend="matplotlib",
    post_process=space_with_overlays
)


page = SolaraViz(
    model, 
    components=[
        make_space_component(agent_portrayal=agent_draw, backend="matplotlib", post_process=space_with_overlays),
        
        #animal_space,

        make_time_plot(["Deer"], post_process=apply_colours),
        make_time_plot([model.predator], post_process=apply_colours),
        # make_time_plot(["Deer Population Normalised", "Wolf Population Normalised"], post_process=apply_colours),

        # make_plot_component(["Total Saplings", "Total Trees"], post_process=apply_colours),
        #make_plot_component(["Deer Hunted", "Total Deer Deaths"]),
        
        # Wolf deaths and energy
        #make_plot_component(["Total Wolf Deaths"]),
        # make_plot_component[("Total Wolf Energy")],
        #make_plot_component(["Mean Wolf Energy"]),
        # Packs
        #make_plot_component(["Number of Packs"]),
        #make_plot_component(["Mean Pack Size"]),


        # make_plot_component(["Deer", model.predator], post_process=apply_colours),
        #make_time_plot(["Deer Population Normalised", "Wolf Population Normalised"], post_process=apply_colours),

        # make_plot_component(["Total Saplings", "Total Trees"], post_process=apply_colours),
        make_time_plot(["Deer Hunted", "Total Deer Deaths"]),
        
        # Wolf deaths and energy
        make_time_plot(["Total Wolf Deaths"]),
        # make_plot_component[("Total Wolf Energy")],
        #make_time_plot(["Mean Wolf Energy"]),
        # Packs
        # make_time_plot(["Number of Packs"]),
        # make_time_plot(["Mean Pack Size"]),

    ],
    model_params={"init_predators": Slider("Initial predators", 10, 1, 20, 1),
        "init_deer": Slider("Initial deer", 100, 1, 1000, 1),},
    name="Species Model",
)

page #noqa


# just run solara run App.py to run 
# need to ensure solara, mesa, networkx and altair are installed to run visualisation