import solara
import matplotlib.pyplot as plt
from mesa.visualization import Slider, SolaraViz, make_space_component, make_plot_component
from Model import SpeciesModel
from VegetationClass import Vegetation

AGENT_COLOURS = {
    "Deer": "orange",
    "Wolf": "blue",
    "Lynx": "brown",
    "Sapling": "#a2c399",
    "Tree": "#5e8354",
}

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


# Initiate the model
model = SpeciesModel(
    max_steps=3000, 
    use_pack_dynamics=True
    )

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

# Create space components
elevation_space_component = make_space_component(
    agent_portrayal=agent_draw,
    backend="matplotlib",
    post_process=space_with_overlays
)

basic_space_component = make_space_component(
    agent_portrayal=agent_draw,
    backend="matplotlib"
)

page = SolaraViz(
    model, 
    components=[
        make_space_component(agent_portrayal=agent_draw, backend="matplotlib", post_process=space_with_overlays),
        
        make_plot_component(["Deer", model.predator], post_process=apply_colours),
        make_plot_component(["Total Saplings", "Total Trees"], post_process=apply_colours),
        make_plot_component(["Deer Hunted", "Total Deer Deaths"]),
        
        # Wolf deaths and energy
        make_plot_component(["Total Wolf Deaths"]),
        # make_plot_component[("Total Wolf Energy")],
        make_plot_component(["Mean Wolf Energy"]),
        # Packs
        make_plot_component(["Number of Packs"]),
        make_plot_component(["Mean Pack Size"]),



    ],
    model_params={"init_predators": Slider("Initial predators", 10, 1, 20, 1),
        "init_deer": Slider("Initial deer", 100, 1, 1000, 1),},
    name="Species Model",
)

page #noqa


# just run solara run App.py to run 
# need to ensure solara, mesa, networkx and altair are installed to run visualisation