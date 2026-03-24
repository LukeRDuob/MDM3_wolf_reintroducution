import solara
from mesa.visualization import Slider, SolaraViz, make_space_component, make_plot_component
from Model import SpeciesModel


def agent_draw(agent):
    """Display the human agents as black dots and zombies as red dots."""
    if agent.species == "Deer":
        return {"color": "orange", "size": 5}
    elif agent.species == "Wolf":
        return {"color": "gray", "size": 5}
    elif agent.species == "Lynx":
        return {"color": "brown", "size": 5}
    

# Initiate the model
model = SpeciesModel(use_pack_dynamics=True)


def space_with_elevation(ax):
    # Draw elevation as background
    ax.imshow(
        model.elevation_grid,
        extent=[0, model.width, 0, model.height],
        origin="lower",
        cmap="terrain",
        alpha=0.6  # transparency so agents show
    )

# Create space components
elevation_space_component = make_space_component(
    agent_portrayal=agent_draw,
    backend="matplotlib",
    post_process=space_with_elevation
)

basic_space_component = make_space_component(
    agent_portrayal=agent_draw,
    backend="matplotlib"
)

page = SolaraViz(
    model, 
    components=[
        make_space_component(agent_portrayal=agent_draw, backend="matplotlib"),
        make_plot_component(["Deer", model.predator])
        ],
    model_params={"init_predators": Slider("Initial predators", 10, 1, 20, 1),
        "init_deer": Slider("Initial deer", 100, 1, 1000, 1),},
    name="Species Model",
)

page #noqa


# just run solara run App.py to run 
# need to ensure solara, mesa, networkx and altair are installed to run visualisation