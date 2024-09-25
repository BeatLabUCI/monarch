import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import os
import pathlib
from .utils import get_valve_events


def pv_loop(model, x_lim=None, y_lim=None, compartments=(2, ), legend=None, cmap="cubehelix", fig_size=(6.0, 4.5),
            show_fig=True, file_type="pdf", file_path=None, file_name="pvloop"):
    """Plot pressure-volume loops from a simulation file"""

    # Colormap
    n_compartments = len(compartments)
    cmap = sns.color_palette(cmap, n_compartments, as_cmap=False)

    # Plot pressure-volume loops
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=fig_size)
    for i, compartment in enumerate(compartments):
        ax.plot(model.volumes[:, compartment], model.pressures[:, compartment], color=cmap[i], linewidth=3)

    # Axes
    plt.xlabel("Volume (mL)", fontsize=12)
    plt.ylabel("Pressure (mmHg)", fontsize=12)

    # Set lower y limit to 0
    ax.set_ylim(bottom=0)

    # Add legend
    if legend:
        ax.legend(compartment_nmbrs_to_names(model, compartments), title="", frameon=False)

    finish_plot(fig, file_path, file_name, file_type, show_fig, x_lim=x_lim, y_lim=y_lim)


def pv_loops(models, x_lim=None, y_lim=None, compartment=2, legend=True, cmap="cubehelix", fig_size=(6.0, 4.5),
             show_fig=True, file_type="pdf", file_path=None, file_name="pvloops", model_names=()):
    """Plot pressure-volume loops from multiple simulation files"""

    if cmap == "cubehelix":
        cmap = sns.cubehelix_palette(as_cmap=False, n_colors=len(models))
    else:
        cmap = sns.color_palette(cmap, as_cmap=False, n_colors=len(models))

    # Plot pressure-volume loops
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=fig_size)
    for i, model in enumerate(models):
        ax.plot(model.volumes[:, compartment], model.pressures[:, compartment], color=cmap[i], linewidth=3)

    # Axes
    plt.xlabel("Volume (mL)", fontsize=12)
    plt.ylabel("Pressure (mmHg)", fontsize=12)

    # Set lower y limit to 0
    ax.set_ylim(bottom=0)

    # Add legend
    if legend and (len(model_names) > 0):
        ax.legend(model_names, title="", frameon=False)

    finish_plot(fig, file_path, file_name, file_type, show_fig, x_lim=x_lim, y_lim=y_lim)


def pressures_volumes(model, p_lim=None, v_lim=None, compartments=(0, 1, 2, 3, 4, 5, 6, 7), fig_size=(6.0, 4.5),
                      legend=True, cmap="cubehelix", show_fig=True, pressures=True, volumes=True,
                      file_type="pdf", file_path=None, file_name="volumes_pressures"):
    """Plot pressures and volumes over time"""

    # Colormap
    n_compartments = len(compartments)
    cmap = sns.color_palette(cmap, n_compartments, as_cmap=False)

    # Plot pressure-volume loops
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, pressures+volumes, figsize=(fig_size[0]*pressures + fig_size[0]*volumes, fig_size[1]))

    # Turn axes into array with at least one dimension
    if pressures and not volumes or not pressures and volumes:
        ax = [ax]

    # Volumes
    if volumes:
        for i, compartment in enumerate(compartments):
            ax[0].plot(model.time, model.volumes[:, compartment], color=cmap[i], linewidth=3)
        ax[0].set_ylabel("Volume (mL)", fontsize=12)
        if v_lim:
            ax[0].set_ylim(v_lim)

    # Pressures
    if pressures:
        for i, compartment in enumerate(compartments):
            ax[-1].plot(model.time, model.pressures[:, compartment], color=cmap[i], linewidth=3)
        ax[-1].set_ylabel("Pressure (mmHg)", fontsize=12)
        if p_lim:
            ax[-1].set_ylim(p_lim)

    # Axes
    for i in range(len(ax)):
        ax[i].set_xlabel("Time (s)", fontsize=12)
        ax[i].set_ylim(bottom=0)

    # Add legend
    if legend:
        ax[-1].legend(compartment_nmbrs_to_names(model, compartments), title="", frameon=False)

    finish_plot(fig, file_path, file_name, file_type, show_fig, x_lim=(model.time[0], model.time[-1]))


def stretch(model, y_lim=None, legend=True, cmap="cubehelix", fig_size=(6.0, 4.5),
            show_fig=True, file_type="pdf", file_path=None, file_name="strain",
            walls=(0, 1, 2), wall_avg=True, shortening=False):
    """Plot strain vs. time from a simulation file"""

    # Shortcuts
    i_ventricles = model.heart.i_ventricles
    n_ventricles = np.sum(i_ventricles)

    # Normalize based on ED frame if shortening
    if shortening:
        lab_f = model.heart.lab_f / model.heart.lab_f[model.outputs["IED"], :] - 1
    else:
        lab_f = model.heart.lab_f

    # Colormap
    if wall_avg:
        wall_colors = sns.color_palette(cmap, as_cmap=False, n_colors=3)
        colors = np.zeros((n_ventricles, 3))
        for i in range(n_ventricles):
            colors[i, :] = wall_colors[model.heart.patches[i]]
    else:
        # Normalized ventricular activation times
        t_norm = ((model.heart.t_act[i_ventricles] - np.min(model.heart.t_act[i_ventricles])) /
                  (np.max(model.heart.t_act[i_ventricles]) - np.min(model.heart.t_act[i_ventricles])))
        if cmap == "cubehelix":
            cmap = sns.cubehelix_palette(as_cmap=True)
        else:
            cmap = sns.color_palette(cmap, as_cmap=True)
        colors = cmap(t_norm)
        wall_colors = None

    # Calculate average wall stretch
    lab_mean = np.zeros((model.time.shape[0], 3))
    for i in range(3):
        lab_mean[:, i] = np.mean(lab_f[:, model.heart.patches == i], axis=1)

    # Start figure
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=fig_size)

    if wall_avg:
        line_styles = ["-", "-", "-"]
    else:
        line_styles = ["-", "--", ":"]
    wall_names = ["LFW", "RFW", "SW"]

    # Patches
    for i in range(n_ventricles):
        if model.heart.patches[i] in walls:
            ax.plot(model.time, lab_f[:, i], color=colors[i], alpha=1 - wall_avg*0.7,
                    linewidth=2, linestyle=line_styles[model.heart.patches[i]], label="_exclude_legend")

    # Walls
    if wall_avg:
        p = []
        for i in range(3):
            p.append(ax.plot(model.time, lab_mean[:, i], color=wall_colors[i], linewidth=4, linestyle=line_styles[i],
                     label=wall_names[i]))

        # Add legend (reduce line width of each entry)
        if legend:
            leg = ax.legend(title="Walls", frameon=False)

    # Axes
    plt.xlabel("Time (s)", fontsize=12)
    if shortening:
        plt.ylabel("Shortening (-)", fontsize=12)
    else:
        plt.ylabel("Stretch (-)", fontsize=12)

    finish_plot(fig, file_path, file_name, file_type, show_fig, x_lim=(model.time[0], model.time[-1]), y_lim=y_lim)


def wiggers_diagram(model, cmap="cubehelix", show_fig=True, file_type="pdf", file_path=None, file_name="wiggers",
                    fig_size=(6.4, 4.8)):
    """Plot Wiggers diagram from a simulation file"""

    # Colormap
    n_compartments = 4
    cmap = sns.color_palette(cmap, n_compartments, as_cmap=False)

    # Plot Wiggers diagram
    sns.set_theme(style="white")
    fig, ax = plt.subplots(3, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1.5, 0.2]})

    # Pressure plot
    ax[0].plot(model.time, model.pressures[:, 2], color=cmap[0], linewidth=3, label="LV")
    ax[0].plot(model.time, model.pressures[:, 1], color=cmap[1], linewidth=3, label="LA")
    ax[0].plot(model.time, model.pressures[:, 3], color=cmap[2], linewidth=3, label="Aortic")
    ax[0].set_ylim(bottom=0)

    # Volume plot
    ax[1].plot(model.time, model.volumes[:, 2], color=cmap[3], linewidth=3, label="LV")

    # Plot valve events
    time_events = get_valve_events(model)
    mv_color = "#888888"
    av_color = "#555555"
    for axi in ax:
        axi.axvline(x=model.time[time_events["mv_closes"]], c=mv_color, linestyle="--", ymin=-0.5, ymax=1.0, linewidth=2, zorder=0, clip_on=False)
        axi.axvline(x=model.time[time_events["av_opens"]], c=av_color, linestyle=":", ymin=-0.5, ymax=1.0, linewidth=2, zorder=0, clip_on=False)
        axi.axvline(x=model.time[time_events["av_closes"]], c=av_color, linestyle="--", ymin=-0.5, ymax=1.0, linewidth=2, zorder=0, clip_on=False)
        axi.axvline(x=model.time[time_events["mv_opens"]], c=mv_color, linestyle=":", ymin=-0.5, ymax=1.0, linewidth=2, zorder=0, clip_on=False)

    # Axes
    ax[0].set_ylabel("Pressure (mmHg)", fontsize=12)
    ax[1].set_xlabel("Time (s)", fontsize=12)
    ax[1].set_ylabel("Volume (mL)", fontsize=12)

    # Add annotations
    ax[0].annotate("Ventricular pressure", xy=(1.025*model.time[-1], model.pressures[-1, 2] - 3),
                   fontsize=12, ha="left", annotation_clip=False, color=cmap[0])
    ax[0].annotate("Atrial pressure", xy=(1.025*model.time[-1], model.pressures[-1, 1] + 3),
                   fontsize=12, ha="left", annotation_clip=False, color=cmap[1])
    ax[0].annotate("Aortic pressure", xy=(1.025*model.time[-1], model.pressures[-1, 3]),
                   fontsize=12, ha="left", annotation_clip=False, color=cmap[2])
    ax[1].annotate("Ventricular volume", xy=(1.025*model.time[-1], model.volumes[-1, 2]),
                   fontsize=12, ha="left", annotation_clip=False, color=cmap[3])

    ax[0].annotate("MV closes", xy=(model.time[time_events["mv_closes"]], ax[0].get_ylim()[1]),
                   fontsize=12, ha="left", annotation_clip=False, color=mv_color, rotation=45)
    ax[0].annotate("AV opens", xy=(model.time[time_events["av_opens"]], ax[0].get_ylim()[1]),
                   fontsize=12, ha="left", annotation_clip=False, color=av_color, rotation=45)
    ax[0].annotate("AV closes", xy=(model.time[time_events["av_closes"]], ax[0].get_ylim()[1]),
                   fontsize=12, ha="left", annotation_clip=False, color=av_color, rotation=45)
    ax[0].annotate("MV opens", xy=(model.time[time_events["mv_opens"]], ax[0].get_ylim()[1]),
                   fontsize=12, ha="left", annotation_clip=False, color=mv_color, rotation=45)

    # Hides all spines except for the left one
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].xaxis.set_visible(False)
        ax[i].set_xlim(model.time[0], model.time[-1])

    # Show spine and ticks for bottom plot
    ax[2].spines['left'].set_visible(False)
    ax[2].yaxis.set_visible(False)
    ax[2].spines['bottom'].set_visible(True)
    ax[2].xaxis.set_visible(True)

    finish_plot(fig, file_path, file_name, file_type, show_fig)


def plot_growth(model, outputs, units="", show_fig=True, file_type="pdf", file_path=None, file_name="growth",
                ax_size=(4.0, 3.0), max_cols=3):
    """Plot output of choice during growth"""

    # Determine how many rows are needed based on max_cols
    n_outputs = len(outputs)
    n_cols = min(n_outputs, max_cols)
    n_rows = int(np.ceil(n_outputs / n_cols))

    # Make list if var is a string
    if isinstance(outputs, str):
        outputs = [outputs]
    if isinstance(units, str):
        units = [units]

    sns.set_theme(style="white")
    fig = plt.figure(figsize=(ax_size[0] * n_cols, ax_size[1] * n_rows))

    for i, (output, unit) in enumerate(zip(outputs, units)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.plot(model.growth.time, model.growth.outputs.loc[:, output], linewidth=3)

        ax.set_ylabel(output + " (" + unit + ")", fontsize=12)
        ax.set_xlabel("Time (days)", fontsize=12)

        ax.set_xlim(model.growth.time[0], model.growth.time[-1])

    finish_plot(fig, file_path, file_name, file_type, show_fig)


def plot_fg(model, walls=(0, 1, 2), show_fig=True, file_type="pdf", file_path=None, file_name="growth",
                     ax_size=(4.0, 3.0), cmap="cubehelix"):

    n_patches = model.heart.n_patches_tot

    jg = np.zeros((model.growth.n_g, n_patches))
    for i_p in range(n_patches):
        for i_g in range(model.growth.n_g):
            jg[i_g, i_p] = np.prod(model.growth.f_g[i_g, :, i_p])

    cmap = sns.color_palette(cmap, n_patches, as_cmap=False)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(len(walls), 4, figsize=(ax_size[0] * 4, ax_size[1] * len(walls)))

    for i_p in range(n_patches):
        if model.heart.patches[i_p] in walls:
            for i in range(3):
                ax[walls.index(model.heart.patches[i_p]), i].plot(model.growth.time, model.growth.f_g[:, i, i_p],
                                                                  color=cmap[i_p], linewidth=3)
            ax[walls.index(model.heart.patches[i_p]), 3].plot(model.growth.time, jg[:, i_p], color=cmap[i_p],
                                                              linewidth=3)

    for i in range(3):
        ax[i, 0].set_ylabel(r"$F_{g,11}$ (-)", fontsize=12)
        ax[i, 1].set_ylabel(r"$F_{g,22}$ (-)", fontsize=12)
        ax[i, 2].set_ylabel(r"$F_{g,33}$ (-)", fontsize=12)
        ax[i, 3].set_ylabel(r"$J_g$ (-)", fontsize=12)
    for i in range(4):
        ax[-1, i].set_xlabel("Time (days)", fontsize=12)

    for i in range(len(walls)):
        for j in range(4):
            ax[i, j].set_xlim(model.growth.time[0], model.growth.time[-1])

    finish_plot(fig, file_path, file_name, file_type, show_fig)


def plot_jg(model, walls=(0, 1, 2), mean=False, show_fig=True, file_type="pdf", file_path=None, file_name="growth",
            ax_size=(4.0, 3.0), cmap="cubehelix"):

    n_patches = model.heart.n_patches_tot

    jg = np.zeros((model.growth.n_g, n_patches))
    for i_p in range(n_patches):
        for i_g in range(model.growth.n_g):
            jg[i_g, i_p] = np.prod(model.growth.f_g[i_g, :, i_p])

    # Calculate mean for all patches with the same value
    jg_mean = np.zeros((model.growth.n_g, len(walls)))
    if mean:
        for i in range(len(walls)):
            jg_mean[:, i] = np.mean(jg[:, model.heart.patches == walls[i]], axis=1)

    sns.set_theme(style="white")

    if mean:
        cmap = sns.color_palette(cmap, len(walls), as_cmap=False)
        fig, ax = plt.subplots(figsize=(ax_size[0], ax_size[1]))
        for i in range(len(walls)):
            ax.plot(model.growth.time, jg_mean[:, i], color=cmap[i], linewidth=3)
        ax.set_ylabel(r"$J_g$ (-)", fontsize=12)
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_xlim(model.growth.time[0], model.growth.time[-1])
        ax.legend(["Wall: " + str(wall) for wall in walls], title="", frameon=False)

    else:
        cmap = sns.color_palette(cmap, n_patches, as_cmap=False)
        fig, ax = plt.subplots(1, len(walls), figsize=(ax_size[0] * len(walls), ax_size[1]))
        for i_p in range(n_patches):
            if model.heart.patches[i_p] in walls:
                ax[walls.index(model.heart.patches[i_p])].set_title("Wall: " + str(model.heart.patches[i_p]))
                ax[walls.index(model.heart.patches[i_p])].plot(model.growth.time, jg[:, i_p], color=cmap[i_p], linewidth=3)

        # Only first plot
        ax[0].set_ylabel(r"$J_g$ (-)", fontsize=12)

        # All plots
        for i in range(len(walls)):
            ax[i].set_xlabel("Time (days)", fontsize=12)
            ax[i].set_xlim(model.growth.time[0], model.growth.time[-1])

    finish_plot(fig, file_path, file_name, file_type, show_fig)


def pv_loop_growth(model, show_fig=True, file_type="pdf", file_path=None, file_name="pvloop_growth",
                   fig_size=(6.0, 4.5), cmap="cubehelix", index=None, legend=True, x_lim=None, y_lim=None):
    """Plot pressure-volume loops of chambers of choice during growth"""

    if index is None:
        if cmap == "cubehelix":
            cmap = sns.cubehelix_palette(as_cmap=False, n_colors=model.growth.n_g)
        else:
            cmap = sns.color_palette(cmap, as_cmap=False, n_colors=model.growth.n_g)
    else:
        cmap = sns.color_palette(cmap, len(index), as_cmap=False)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=fig_size)

    if index is None:
        for i_g in range(model.growth.n_g):
            ax.plot(model.growth.v_lv[i_g, :], model.growth.p_lv[i_g, :], color=cmap[i_g], linewidth=3)
    else:
        for i, i_g in enumerate(index):
            ax.plot(model.growth.v_lv[i_g, :], model.growth.p_lv[i_g, :], color=cmap[i], linewidth=3)

    ax.set_ylabel("Pressure (mmHg)", fontsize=12)
    ax.set_xlabel("Volume (mL)", fontsize=12)

    # Set lower y limit to 0
    ax.set_ylim(bottom=0)

    if legend and index is not None:
        ax.legend(["t = " + str(model.growth.time[i]) + "days" for i in index], title="", frameon=False)

    finish_plot(fig, file_path, file_name, file_type, show_fig, x_lim=x_lim, y_lim=y_lim)


def finish_plot(fig, file_path, file_name, file_type, show_fig, set_box2=True, x_lim=None, y_lim=None):
    """Utility function to show and/or save figure; set limits; and/or set box lines"""

    # Tight layout
    fig.tight_layout()

    # Get axes
    axs = fig.get_axes()

    if set_box2:
        for ax in axs:
            set_box_lines(ax)

    if x_lim:
        for ax in axs:
            ax.set_xlim(left=x_lim[0], right=x_lim[1])
    if y_lim:
        for ax in axs:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

    # Save figure
    if file_path and file_name is not None:

        # If file_path is a string, convert to pathlib.Path
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        # Create file path if not existing
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(file_path / str(file_name + "." + file_type), bbox_inches="tight")

    # Show figure, or close it
    if show_fig:
        plt.show()
    else:
        plt.close()


def set_box_lines(ax):
    # Set axes thickness to 2
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)


def compartment_nmbrs_to_names(model, compartment_nmbrs):
    """Convert compartment numbers to names"""

    return [model.compartment_names[compartment] for compartment in compartment_nmbrs]


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def wide2long(data, index, columns):
    # Convert single variable numpy array vs time to long-format pandas format (e.g. stretch vs. time)
    df0 = pd.DataFrame(data, index=index, columns=columns)
    df = pd.DataFrame(df0, columns=columns).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})
    return df


def wide2long_2vars(data1, data2, index, columns):
    # Convert two numpy arrays vs time to single long-format pandas format (e.g. pressure vs. volume)

    df0 = pd.DataFrame(data1, index=index, columns=columns)
    df = pd.DataFrame(df0, columns=columns).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})

    df0_2 = pd.DataFrame(data2, index=index, columns=columns)
    df_2 = pd.DataFrame(df0_2, columns=columns).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})

    df.insert(3, "value2", df_2['value'])
    return df