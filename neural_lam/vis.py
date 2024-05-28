# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# First-party
from neural_lam import constants, utils

def plot_error_curves(errors, dataset_constants, title=None, step_length=6, summary=False):
    """
    Plot error curves for different variables at different
    prediction horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    var_subset = dataset_constants.HEAT_MAP_VARS if summary else range(errors_np.shape[0])
    d_f, pred_steps = errors_np.shape
    
    figs = []
    for var_idx in var_subset:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors_np[var_idx, :], marker="o")
        
        ax.set_xticks(np.arange(pred_steps))
        pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
        pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
        ax.set_xticklabels(pred_hor_h)
        
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("Normalized error")
        ax.set_title(dataset_constants.PARAM_NAMES_SHORT[var_idx])
        # ax.legend()
        
        if title:
            ax.set_title(title)
        
        var_name = dataset_constants.PARAM_NAMES_SHORT[var_idx]
        var_climatology = dataset_constants.CLIMATOLOGY.get(var_name)
        if var_climatology:
            ax.axhline(var_climatology, color="grey", linestyle="--")

        figs.append((var_name, fig))

    return figs

@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, dataset_constants, title=None, step_length=6, summary=False):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    var_subset = dataset_constants.HEAT_MAP_VARS if summary else range(errors_np.shape[0])
    errors_np = errors_np[var_subset, :]
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(30, 20))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{dataset_constants.PARAM_NAMES_SHORT[var_idx]} ({dataset_constants.PARAM_UNITS[var_idx]})"
        for var_idx in var_subset
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, obs_mask, title=None, vrange=None):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": constants.LAMBERT_PROJ}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
        im = ax.imshow(
            data_grid,
            origin="lower",
            extent=constants.GRID_LIMITS,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8), subplot_kw={"projection": constants.LAMBERT_PROJ}
    )

    ax.coastlines()  # Add coastline outlines
    error_grid = error.reshape(*constants.GRID_SHAPE).cpu().numpy()

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=constants.GRID_LIMITS,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
