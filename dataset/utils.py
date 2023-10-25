import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import f1_score, precision_score, recall_score


def save_fig(save_to, plt=plt, tight=True):
    """
    Save a figure.
    """
    if tight:
        plt.tight_layout()
    plt.savefig(save_to, format="png", dpi=300)
    try:
        plt.cla()
        plt.clf()
        plt.close("all")
    except AttributeError:
        pass


def heatmap_for_cls_report(
    x,
    colour,
    save_to,
    x_label,
    y_label,
    title=None,
    height=15,
    width=10,
    transpose=True,
):
    """
    Create heatmaps per cancer type.
    """
    cmap = sns.color_palette(colour, as_cmap=True)

    print("heatmap:", save_to)

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    divider = make_axes_locatable(ax)
    cbar_ax = (
        divider.new_horizontal(size="5%", pad=0.5, pack_start=False)
        if transpose
        else divider.new_vertical(size="5%", pad=0.5, pack_start=False)
    )

    fig.add_axes(cbar_ax)

    x = x[["precision", "recall", "f1-score"]].sort_values(by="f1-score")
    x = x.transpose() if transpose else x

    ax = sns.heatmap(
        x,
        annot=True,
        linewidth=0.5,
        square=True,
        ax=ax,
        cbar_ax=cbar_ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={"pad": 0.02}
        if transpose
        else {"pad": 0.02, "orientation": "horizontal"},
    )
    ax.set_aspect("equal")

    if not transpose:
        cbar_ax.xaxis.set_ticks_position("top")

    if title is not None:
        ax.set_title(title)

    save_fig(save_to, fig)


def results(y_true, y_pred, accuracy_score, sample_wise=True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accuracy": [accuracy_score(y_true, y_pred)],
            "f1_samples": [f1_score(y_true, y_pred, average="samples")]
            if sample_wise
            else [np.nan],
            "f1_macro": [f1_score(y_true, y_pred, average="macro")],
            "f1_micro": [f1_score(y_true, y_pred, average="micro")],
            "precision_samples": [precision_score(y_true, y_pred, average="samples")]
            if sample_wise
            else [np.nan],
            "precision_macro": [precision_score(y_true, y_pred, average="macro")],
            "precision_micro": [precision_score(y_true, y_pred, average="micro")],
            "recall_samples": [recall_score(y_true, y_pred, average="samples")]
            if sample_wise
            else [np.nan],
            "recall_macro": [recall_score(y_true, y_pred, average="macro")],
            "recall_micro": [recall_score(y_true, y_pred, average="micro")],
        }
    )


def remove_brackets(x: str):
    x = re.sub(r"\(.*?\)", "", x)
    x = re.sub(r"\[.*?\]", "", x)
    x = re.sub(" +", " ", x)
    return x.lower().strip()


def split_treatment(x: str, base_dataset):
    return sorted(
        [
            base_dataset.alternative_name(remove_brackets(y.lower().strip()))
            for y in x.split("+")
        ]
    )


def process_plus(x: str, base_dataset):
    x = split_treatment(x, base_dataset)

    for treatment in base_dataset.all_treatments:
        treatment = split_treatment(treatment, base_dataset)
        if dict.fromkeys(x) == dict.fromkeys(treatment):
            return "".join(treatment)

    return "".join(x)
