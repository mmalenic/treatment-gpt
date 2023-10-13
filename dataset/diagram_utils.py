import matplotlib as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset.load_protect import LoadProtect


def save_fig(save_to, plt=plt, tight=True):
    """
    Save a figure.
    """
    if tight:
        plt.tight_layout()
    plt.savefig(save_to, format="svg", dpi=300)
    try:
        plt.close()
    except AttributeError:
        pass


def add_cancer_types_code(x, from_protect: LoadProtect):
    """
    Add cancer type code to dataframe.
    """
    x["cancer_type_code"] = from_protect.cancer_types.cancer_type_code(x["cancer_type"])
    return x


def heatmaps_per_cancer_type(x, colour, save_to):
    """
    Create heatmaps per cancer type.
    """
    x["cancer_type"] = f"Scores for {x['cancer_type'].iloc[0]}"

    cmap = sns.color_palette(colour, as_cmap=True)
    x = x.set_index("Treatment")

    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(10)

    ax.set_xlabel("Treatment")
    ax.set_ylabel("Score")

    divider = make_axes_locatable(ax)
    cbar_ax = divider.new_horizontal(size="5%", pad=0.5, pack_start=False)
    fig.add_axes(cbar_ax)

    ax = sns.heatmap(
        x[["precision", "recall", "f1-score"]].sort_values(by="f1-score").transpose(),
        annot=True,
        linewidth=0.5,
        square=True,
        ax=ax,
        cbar_ax=cbar_ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={"pad": 0.02},
    )
    ax.set_aspect("equal")

    ax.set_title(x["cancer_type"].iloc[0])

    save_fig(save_to, fig)


def plot_heatmaps(df, save_to):
    """
    Plot all heatmaps.
    """
    df.groupby(["cancer_type"]).apply(
        lambda x: heatmaps_per_cancer_type(
            x, "Blues", f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_blue.svg"
        )
    )
    df.groupby(["cancer_type"]).apply(
        lambda x: heatmaps_per_cancer_type(
            x, "Reds", f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_red.svg"
        )
    )
    df.groupby(["cancer_type"]).apply(
        lambda x: heatmaps_per_cancer_type(
            x, "Greens", f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_green.svg"
        )
    )
    df.groupby(["cancer_type"]).apply(
        lambda x: heatmaps_per_cancer_type(
            x,
            "Oranges",
            f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_orange.svg",
        )
    )
    df.groupby(["cancer_type"]).apply(
        lambda x: heatmaps_per_cancer_type(
            x,
            "Purples",
            f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_purple.svg",
        )
    )
