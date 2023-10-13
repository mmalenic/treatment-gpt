from matplotlib import pyplot as plt
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


def heatmap_for_cls_report(
    x, colour, save_to, x_label, y_label, title, height=15, width=10
):
    """
    Create heatmaps per cancer type.
    """
    cmap = sns.color_palette(colour, as_cmap=True)

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

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

    ax.set_title(title)

    save_fig(save_to, fig)
