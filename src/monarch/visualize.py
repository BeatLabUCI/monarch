import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import os

import utils as utils


def pressures_volumes(file_path, file_name, save_name="pv_loop", x_lim=None, y_lim=None, compartment=[2], legend=None, save_fig=True, show_fig=False, file_type="pdf"):
    """
    pressures_volumes plots pressure-volume loops from chosen model compartments
    
    :param file_path: Path to the file that stores the exported plots
    :param file_name: Name of the file that stores the exported plots
    :param save_name: Name of the save for the exported file
    :param x_lim: The x bounds of the plot
    :param y_lim: The y bounds of the plot
    :param compartment: Compartment of the heart being graphed
    :param legend: Legend for the plot
    :param save_fig: Should the figure be exported to a file
    :param show_fig: Should the figure be shown in a popup window
    :param file_type: What type of file should the exported plots be saved in
    """

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f["time"][:]*1e3
        volumes = f["volumes"][:, compartment]
        pressures = f["pressures"][:, compartment]

    if not legend:
        legend = compartment
        legend_switch = False
    else:
        legend_switch = True

    # Pandas format
    pv_df = pd.DataFrame()
    for i in range(np.size(volumes, 1)):
        pv_df_i = wide2long_2vars(volumes[:, i], pressures[:, i], time, [legend[i]])
        pv_df = pd.concat([pv_df, pv_df_i], ignore_index=True)

    # PV loop
    f, ax = plt.subplots()
    g = sns.lineplot(data=pv_df, x="value", y="value2", hue="variable", legend=legend_switch, errorbar=None, sort=False)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    if x_lim: plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])

    # Remove legend title
    if legend_switch: g.legend_.set_title(None)

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def stress_stretch(file_path, file_name, x_lim=None, y_lim=None, save_name="stress_stretch", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        lab_f = f["lab_f"][:, 0:21]
        sig_f = f["sig_f"][:, 0:21]
        time = f["time"][:]*1e3
        patches = f.attrs['patches'][0:21]

    # Convert wall numbers to wall names
    walls = ["Lateral wall", "Right free wall", "Septal wall"]
    patches = [walls[x] for x in patches]

    # Convert to pandas format
    lab_sig_f = wide2long_2vars(lab_f[::10,:], sig_f[::10,:]*1e3, time[::10], patches)

    # Plot
    f, ax = plt.subplots()
    ax = sns.lineplot(data=lab_sig_f, x="value", y="value2", hue="variable", palette="rocket", sort=False)
#    plt.ylim(bottom=y_lim[0], top=y_lim[1])
    if x_lim: plt.xlim(left =time[0], right=time[-1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.xlabel("Stretch (-)")
    plt.ylabel("Cauchy stress (kPa)")
    ax.get_legend().set_title("")

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def wiggers(file_path, file_name, x_lim=None, y_lim=None, save_name="wiggers", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f["time"][:]*1e3
        volumes = f["volumes"][:, 1:4]
        pressures = f["pressures"][:, 1:4]

    compartments = ["Atrial", "Ventricular", "Arterial"]

    # Pandas format
    pv_df = wide2long_2vars(volumes, pressures, time, compartments)

    # PV loop
    f, ax = plt.subplots()
    sns.lineplot(data=pv_df.loc[pv_df['variable'] == 'Ventricular'], x="value", y="value2",  hue="variable", legend=False, errorbar=None, sort=False)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    if x_lim: plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.savefig(file_path + "/" + file_name + '_pv_loop.pdf', bbox_inches='tight')

    # Wiggers diagram
    fig, ax = plt.subplots(2,1, figsize=(10,8))

    sns.lineplot(data=pv_df, ax=ax[0], x="Time", y="Pressure",  hue="variable", legend=False, errorbar=None, sort=False)
    plt.ylim(bottom=y_lim[0])

    sns.lineplot(data=pv_df.loc[pv_df['variable'] == 'Ventricular'], ax=ax[1], x="Time", y="value",  hue="variable", legend=False, errorbar=None, sort=False)

    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def pv_growth(file_path, file_name, x_lim=None, y_lim=None, compartment="lv", save_name="pv_loop", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        v_lv = f["v_" + compartment][:]
        p_lv = f["p_" + compartment][:]
        tg = f.attrs['outputs_rows'][:]

    # Pandas format
    pv_df0 = pd.DataFrame(np.transpose(v_lv), index=np.arange(0,v_lv.shape[1]), columns=tg)
    pv_df = pd.DataFrame(pv_df0, columns=tg).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})
    p_df0 = pd.DataFrame(np.transpose(p_lv), index=np.arange(0,p_lv.shape[1]), columns=tg)
    p_df = pd.DataFrame(p_df0, columns=tg).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})
    pv_df.insert(3, "Pressure", p_df['value'])

    sns.lineplot(data=pv_df, x="value", y="Pressure",  hue="variable", legend=False, errorbar=None, sort=False)

    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    if x_lim: plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

def growth_volumes(file_path, file_name, save_name="edv_esv", save_fig=True, show_fig=False, file_type="pdf"):

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:

        tg = f.attrs['outputs_rows'][:]
        outputs_columns = f.attrs['outputs_columns'][:]
        outputs = f['outputs'][:]
        f_g = f["f_g"][:]
        s_l = f["s_l"][:]

    edv = outputs[:, outputs_columns == "LVEDV"]
    esv = outputs[:, outputs_columns == "LVESV"]
    rvedv = outputs[:, outputs_columns == "RVEDV"]
    rvesv = outputs[:, outputs_columns == "RVESV"]

    plt.plot(tg, edv, linewidth=2)
    plt.plot(tg, esv, linewidth=2)
    plt.ylabel("LV Volume (mL)")
    plt.xlabel("Days")
    plt.show()

    plt.plot(tg, rvedv)
    plt.plot(tg, rvesv)
    plt.ylabel("Volume (mL)")
    plt.xlabel("Days")
    plt.show()

    plt.plot(tg, f_g[:, 0, 0])
    plt.plot(tg, f_g[:, 1, 0])
    plt.plot(tg, f_g[:, 2, 0])
    plt.ylabel("Fiber Growth")
    plt.xlabel("Days")
    plt.show()

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type))
    if show_fig:
        plt.show()
    else:
        plt.close()


def mass(file_path, file_name, save_name="wallmass", x_lim=None, y_lim=None, save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        f_g = f["f_g"][:]
        time = f.attrs['outputs_rows'][:]

    # Walls [lat-sep-pos-ant]
    n_patches = 16
    patches = [[5, 6, 12, 13, 16], [2, 3, 8, 9, 14], [4, 10, 15], [1, 7, 13]]
    titles = ["Lateral", "Septal", "Posterior", "Anterior"]

    # Get relative patch mass change from the growth tensor
    n_g = f_g.shape[0]
    wall_mass = np.zeros((n_g, n_patches))
    for i_g in range(n_g):
        for patch in range(n_patches):
            wall_mass[i_g, patch] = np.prod(f_g[i_g,:,patch])
    wall_mass = pd.DataFrame(wall_mass, index=time, columns=np.arange(1,17))

    # Set up plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    sns.set_context("notebook")
    # Lateral wall
    for ax, i_wall in zip(ax.ravel(), range(4)):
        ax.set_title(titles[i_wall])
        ax.set_box_aspect(0.8)
        df = pd.DataFrame(wall_mass[patches[i_wall]], columns=patches[i_wall]).reset_index().melt(id_vars = 'index')\
             .rename(columns={'index':'Time'})

        sns.lineplot(data=df, x="Time", y="value", ax=ax, palette="dark", linewidth=3)
        sns.lineplot(data=df, x="Time", y="value",  hue="variable", ax=ax)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        if x_lim: ax.set_xlim(bottom=x_lim[0], top=x_lim[1])
        if y_lim: ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        ax.legend(loc='upper center', ncol=3, )
        plt.setp(ax.get_legend().get_texts(), fontsize='8')

    fig.add_subplot(111, frame_on=False)
    plt.tight_layout()
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel('Wall mass change (-)')
    plt.xlabel('Time (days)')

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_growth(file_path, file_name, save_name="growth", save_fig=True, show_fig=False,
                file_type="pdf", cmap="cubehelix", n_patches=None):
    """Plot growth tensor components over time"""

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        f_g = f["f_g"][:]
        time = f.attrs['outputs_rows'][:]

    if not n_patches:
        n_patches = f_g.shape[2]

    cmap = sns.color_palette(cmap, n_patches, as_cmap=False)

    # Set up plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    sns.set_context("notebook")

    titles = ["Fiber", "Cross-fiber", "Radial"]

    for i_g in range(2):
        ax[i_g].set_title(titles[i_g])
        ax[i_g].set_box_aspect(0.8)
        for i_patch in range(n_patches):
            ax[i_g].plot(time, f_g[:, i_g*2, i_patch], linewidth=3, color=cmap[i_patch])

        ax[i_g].set(xlabel="Time (days)", ylabel="Growth (-)")
        ax[i_g].set_ylim(bottom=np.floor(np.min(f_g[:, i_g*2, :])* 10) / 10,
                    top=np.ceil(np.max(f_g[:, i_g*2, :])* 10) / 10)

        ax[i_g].set_xlim(left=np.min(time), right=np.max(time))

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_growth_stimuli(file_path, file_name, save_name="growth_stimuli", save_fig=True, show_fig=False,
                file_type="pdf", cmap="cubehelix", n_patches=None):
    """Plot growth tensor components over time"""

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        s_l = f["s_l"][:]
        s_r = f["s_r"][:]
        f_g = f["f_g"][:]
        time = f.attrs['outputs_rows'][:]

    if not n_patches:
        n_patches = f_g.shape[2]

    cmap = sns.color_palette(cmap, n_patches, as_cmap=False)

    # Set up plot
    fig, ax = plt.subplots(1, 2, figsize=(10*2/3, 4))
    sns.set_context("notebook")

    titles = ["Dilation", "Thickening"]

    for i_g in range(2):
        if i_g == 0:
            stim = s_l
        else:
            stim = s_r
        ax[i_g].set_title(titles[i_g])
        ax[i_g].set_box_aspect(0.8)
        for i_patch in range(n_patches):
            ax[i_g].plot(time, stim[:, i_patch], linewidth=3, color=cmap[i_patch])

        ax[i_g].set(xlabel="Time (days)", ylabel="Growth stimulus (-)")
        ax[i_g].set_ylim(bottom=np.floor(np.min(stim)* 10) / 10,
                    top=np.ceil(np.max(stim)* 10) / 10)

        ax[i_g].set_xlim(left=np.min(time), right=np.max(time))

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_growth_edv(file_path, file_name, save_name="growth_edv", save_fig=True, show_fig=False, file_type="pdf"):
    """Plot EDV over time"""

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f.attrs['outputs_rows'][:]
        outputs_columns = f.attrs['outputs_columns'][:]
        outputs = f['outputs'][:]

    edv = outputs[:, outputs_columns == "LVEDV"]

    plt.plot(time, edv)
    plt.ylabel("LVEDV (mL)")
    plt.xlabel("Time (days)")
    plt.show()

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_growth_edwth(file_path, file_name, save_name="growth_edv", save_fig=True, show_fig=False, file_type="pdf"):
    """Plot EDWth over time"""

    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f.attrs['outputs_rows'][:]
        outputs_columns = f.attrs['outputs_columns'][:]
        outputs = f['outputs'][:]

    edwth_lfw = outputs[:, outputs_columns == "EDWthLfw"]
    edwth_sw = outputs[:, outputs_columns == "EDWthSw"]

    plt.plot(time, edwth_lfw, label="Left free wall")
    plt.plot(time, edwth_sw, label="Septal wall")
    plt.ylabel("EDWth (mm)")
    plt.xlabel("Time (days)")
    plt.legend(frameon=False)
    plt.show()

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def activation(file_path, file_name, color_palette="rocket_r", color_bar=True, norm0=True, save_name="activation", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        data = f['t_act'][0:16]

    if norm0:
        data = data - min(data)

    fig, ax = plt.subplots(figsize=(4, 5))

    # Construct color map
    cmap = sns.color_palette(color_palette, as_cmap=True)
    norm = mpl.colors.Normalize(vmin=min(data), vmax=max(data))

    # Assign color to each segment
    cdata = cmap(norm(data))

    width = 0.28
    wedgeprops = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 2, 'width': width}
    ax.pie(1/6*np.ones(6), wedgeprops=wedgeprops, radius=1, startangle=60, colors=cdata[0:6, :])
    ax.pie(1/6*np.ones(6), wedgeprops=wedgeprops, radius=1-width, startangle=60, colors=cdata[6:12, :])
    ax.pie(1/4*np.ones(4), wedgeprops=wedgeprops, radius=1-2*width, startangle=45, colors=cdata[12:16, :])

    if color_bar:
        axl = fig.add_axes([0.2, 0.1, 0.6, 0.05])
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                     cax=axl, orientation='horizontal', label='Activation timing (ms)')

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def shortening(file_path, file_name, y_lim=None, patches=None,
               save_name="shortening", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        lab_f = f["lab_f"][:, 0:21]
        time = f["time"][:]*1e3
        t_act = f['t_act'][0:21]
        outputs_columns = f.attrs['outputs_names'][:]
        outputs = f['outputs'][0]

    # Get IED
    i_ed = int(outputs[outputs_columns == "IED"][0])

    if patches is None:
        patches = range(lab_f.shape[1])

    # Calculate shortening, and change order so that i_ed is the first element
    shortening = lab_f[:, patches] / lab_f[i_ed, patches] - 1
    shortening = np.roll(shortening, -i_ed, axis=0)

    cmap = sns.cubehelix_palette(shortening.shape[1], as_cmap=False)

    for i in range(shortening.shape[1]):
        plt.plot(time, shortening[:, i], linewidth=3, color=cmap[i])

    plt.ylabel("Shortening (-)")
    plt.xlabel("Time (ms)")
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def stretch_beat(file_path, file_name, y_lim=None, save_name="stretch", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        lab_f = f["lab_f"][:, 0:21]
        time = f["time"][:]*1e3
        t_act = f['t_act'][0:21]
        patches = f.attrs['patches'][0:21]
        pressures = f["pressures"][:]

    # Convert wall numbers to wall names, include only lateral segments
    walls = ["Lateral wall", "Right free wall", "Septal wall"]
    patches = np.array([walls[x] for x in patches])

    # Get valve openings
    valve_events = utils.get_valve_events(pressures)

    # Plot LV wall stretch, exclude ant and post walls
    # pos_ant = [0,3,6,9,12,14]
    sep_lat = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
    plot_stretch_walls(time, lab_f[:, sep_lat], valve_events, patches[sep_lat], t_act[sep_lat], y_lim)
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "_walls." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

    # Plot LV stretch
    plot_stretch(time, lab_f[:,0:16], valve_events, patches[0:16], t_act[0:16], y_lim)
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

    # Plot patch shortening
    plot_shortening(time, lab_f[:,0:16], valve_events, patches[0:16], t_act[0:16], np.array([-0.5,0.5])*(y_lim[1]-y_lim[0]))
    plt.savefig(file_path + "/" + file_name + "_shortening.pdf", bbox_inches='tight')


def strain_beat(file_path, file_name, y_lim=None, save_fig=True, save_name="strain", show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        lab_f = f["lab_f"][:, 0:21]
        time = f["time"][:]*1e3
        t_act = f['t_act'][0:21]
        patches = f.attrs['patches'][0:21]
        pressures = f["pressures"][:]

    # Convert wall numbers to wall names, include only lateral segments
    walls = ["Lateral wall", "Right free wall", "Septal wall"]
    patches = np.array([walls[x] for x in patches])

    # Get valve openings
    valve_events = utils.get_valve_events(pressures)

    # Plot LV wall strain, exclude ant and post walls
    # pos_ant = [0,3,6,9,12,14]
    sep_lat = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]

    # Plot LV strain
    plot_strain(time, lab_f[:, sep_lat], valve_events, patches[sep_lat], t_act[sep_lat], y_lim)

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

def stretch_growth(file_path, file_name, time_frame = 0, y_lim=None, save_name="stretch", save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        lab_f = f["lab_f"][time_frame,:,0:21]
        outputs_columns = f.attrs['outputs_columns'][:]
        outputs = f['outputs'][time_frame]
        t_act = f['t_act'][time_frame,0:21]
        patches = f.attrs['patches'][0:21]

    # Use heart rate to reconstruct time array
    time = np.linspace(0, float(60 / outputs[outputs_columns == "HR"]), lab_f.shape[0]) * 1e3

    # Convert wall numbers to wall names
    # walls = ["Left free wall", "Right free wall", "Septal wall"]
    # patches = [walls[x] for x in patches]

    # Plot LV wall stretch (Resample to plot faster)
    plot_stretch_walls(time[::10], lab_f[::10,:], patches, t_act, y_lim)
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "_walls." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()

    # Plot LV stretch
    plot_stretch(time, lab_f[:,0:16], patches[0:16], t_act[0:16], y_lim)
    plt.savefig("stretch" + str(time_frame) + ".pdf", bbox_inches='tight')
    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_stretch_walls(time, lab_f, valve_events, patches, colors, y_lim=None):

    f, ax = plt.subplots()

    # Convert to long format, resample to plot faster and take out anterior and posterior walls
    lab_f_df = wide2long(lab_f[::10], time[::10], patches)

    # Plot valve timing
    plot_valve_timing(time, valve_events, ax)

    ax = sns.lineplot(data=lab_f_df, x="Time", y="value", hue="variable", palette="rocket")
    plt.xlim(left=time[0], right=time[-1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Stretch (-)")
    ax.get_legend().set_title("")

    return ax


def plot_stretch(time, lab_f, valve_events, patches, colors, y_lim=None):

    colors = [int(x) for x in colors]
    f, ax = plt.subplots()

    # Convert to pandas format
    lab_f_df = wide2long(lab_f, time, colors)
    lab_f_df.insert(0, "Wall", np.repeat(patches,lab_f.shape[0]), True)

    # Plot valve timing
    plot_valve_timing(time, valve_events, ax)

    # Plot stretch
    sns.lineplot(data=lab_f_df, x="Time", y="value", style="Wall",  hue="variable", legend=True, errorbar=None)
    plt.xlim(left=time[0], right=time[-1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Stretch (-)")

    h = plt.gca().get_lines()
    lg = plt.legend(handles=h, labels=["Septal wall", "Left free wall"], loc='best')

    return ax


def plot_shortening(time, lab_f, valve_events, patches, colors, y_lim=None):

    colors = [int(x) for x in colors]
    f, ax = plt.subplots()

    # Normalize with respect to MV closure
    lab_f0 = lab_f[valve_events['mv_closes'], :]
    lab_f = lab_f/lab_f0[np.newaxis, :]-1

    # Convert to pandas format
    lab_f_df = wide2long(lab_f, time, colors)
    lab_f_df.insert(0, "Wall", np.repeat(patches, lab_f.shape[0]), True)

    # Plot valve timing
    plot_valve_timing(time, valve_events, ax)

    # Plot stretch
    sns.lineplot(data=lab_f_df, x="Time", y="value", style="Wall",  hue="variable", legend=False, errorbar=None)
    plt.xlim(left=time[0], right=time[-1])
    if y_lim.size != 0: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Stretch (-)")
    h = plt.gca().get_lines()
    # lg = plt.legend(handles=h, labels=["Septal wall", "Left free wall"], loc='best')

    return ax


def plot_strain(time, lab_f, valve_events, patches, colors, y_lim=None):

    colors = [int(x) for x in colors]
    f, ax = plt.subplots()

    # Calculate Green-Lagrange strain
    eps_f = 0.5*(lab_f**2 - 1)

    # Convert to pandas format
    eps_f_df = wide2long(eps_f, time, colors)
    eps_f_df.insert(0, "Wall", np.repeat(patches,eps_f.shape[0]), True)

    # Plot valve timing
    plot_valve_timing(time, valve_events, ax)

    # Plot stretch
    sns.lineplot(data=eps_f_df, x="Time", y="value", style="Wall",  hue="variable", legend=True, errorbar=None)
    plt.xlim(left=time[0], right=time[-1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Strain (-)")

    h = plt.gca().get_lines()
    lg = plt.legend(handles=h, labels=["Septal wall", "Left free wall"], loc='best')

    return ax


def plot_valve_timing(time, valve_events, ax):
    ax.axvspan(time[valve_events['av_opens']], time[valve_events['av_closes']], alpha=0.1, color='grey', linewidth=None)
    plt.axvline(time[valve_events['av_opens']], alpha=0.3, color='grey', linestyle='-')
    plt.axvline(time[valve_events['av_closes']], alpha=0.3, color='grey', linestyle='-')
    plt.axvline(time[valve_events['mv_closes']], alpha=0.4, color='grey', linestyle='--')
    plt.axvline(time[valve_events['mv_opens']], alpha=0.4, color='grey', linestyle='--')


def mmode(file_path, file_name, save_fig=True, save_name="mmode", show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        pressures = f['pressures'][:]
        wall_thickness = f['wall_thickness'][:, [0, 2]]
        x_m = f['x_m'][:, [0, 2]]

    # Repeats to mimic ultrasound, approximate position of epicardium and endocardium from midwall
    repeats = 7         # (the magic number)
    n = x_m.shape[0]
    x = np.arange(0, repeats*n)
    y_endo = np.tile(x_m - 0.5*np.sign(x_m)*wall_thickness, (repeats, 1))
    y_epi = np.tile(x_m + 0.5*np.sign(x_m)*wall_thickness, (repeats, 1))

    plt.style.use('dark_background')
    fig, ax = plt.subplots(2, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [4, 1]})

    # Plot valve openings
    valve_events = utils.get_valve_events(pressures)
    for i in range(0, repeats):
        ax[0].axvline(valve_events['mv_closes']+i*n, color='#ffffff', linestyle='--', linewidth=1, zorder=1)

    for i in range(0, 2):
        ax[0].fill_between(x, y_endo[:, i], y_epi[:, i], color='#777777', zorder=2)
        ax[0].plot(x, y_endo[:, i], color='#cccccc', linewidth=4, zorder=2)
        ax[0].plot(x, y_epi[:, i], color='#cccccc', linewidth=4, zorder=2)

    # Plot pressure wave
    for i in range(0, repeats):
        ax[0].axvline(valve_events['mv_closes']+i*n, color='#ffffff', linestyle='--', linewidth=1, zorder=1)
    ax[1].plot(x, np.tile(pressures[:, 2], repeats), color='#ffffff')

    # Take care of axes
    ax[0].set_xlim(left=0, right=7*n)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].set_xlim(left=0, right=7*n)
    ax[1].axis('off')

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def geometry_movie(file_path, file_name, slomo=4):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f["time"][:]*1e3
        patches = f.attrs['patches'][:]
        xm = f["x_m"][:]
        r_m = f["r_m"][:]
        lab_f = f["lab_f"][:]

    # Real time (or slowed down by factor slomo)
    fps = len(time)/(time[1]-time[0])/slomo

    # Define the meta data for the movie
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Heart beat', artist='cardiogrowth',
                    comment='Just beat it')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    axlim = xm.max()*1.1

    # Plot frames
    fig, ax = plt.subplots()
    with writer.saving(fig, file_path + "/" + file_name + "_beat.mp4", 100):
        for time_frame in range(0,len(time),1):
            ax.clear()
            plt.xlim(left=-axlim, right=axlim)
            plt.ylim(bottom=-axlim, top=axlim)
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            # geometry_patches(xm[time_frame,:], rm[time_frame,:], lab_f[time_frame,:], patches, ax)
            plot_geometry(xm[time_frame,:], r_m[time_frame,:], ax)
            writer.grab_frame()


def geometry_still(file_path, file_name, time_frame, save_name="geometry", real_wall_thickness=False,
                   save_fig=True, show_fig=False, file_type="pdf"):
    with h5py.File(file_path + "/" + file_name + ".hdf5", "r") as f:
        time = f["time"][:]*1e3
        patches = f.attrs['patches'][:]
        x_m = f["x_m"][time_frame, :]
        r_m = f["r_m"][time_frame, :]
        lab_f = f["lab_f"][time_frame, :]
        wall_thickness = f["wall_thickness"][time_frame, :]

        if not real_wall_thickness:
            wall_thickness = np.array([])

        fig, ax = plt.subplots()
        #geometry_patches(x_m, r_m, lab_f, patches, ax)
        plot_geometry(x_m, r_m, wall_thickness, ax)

        if save_fig:
            plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight')
        if show_fig:
            plt.show()
        else:
            plt.close()


def plot_geometry(xm, rm, wall_thickness, ax, cmap="rocket"):

    alpha = np.linspace(0, 2*np.pi, 1000)

    cmap = sns.color_palette(cmap, 3)
    cmap = [cmap[0],cmap[2], cmap[1]]

    for wall in range(3):

        xp = rm[wall]*np.cos(alpha)
        yp = rm[wall]*np.sin(alpha)
        if rm[wall] > 0:
            ikeep = ((xm[wall]-rm[wall]-xp)>=0)
        elif rm[wall] <= 0:
            ikeep = ((xm[wall]-rm[wall]-xp)<0)

        # Wall arc
        x = xm[wall]-rm[wall]-xp[ikeep]
        y = yp[ikeep]
        if x.size == 0:
            x = np.array([0, 0])
            y = np.array([-1, 1])

        if np.any(wall_thickness):
            # Real wall thickness
            data_linewidth_plot(-x, y, ax=ax, linewidth=wall_thickness[wall], color=cmap[wall], solid_capstyle='round')
        else:
            # Standardized wall thickness
            ax.plot(-x, y, label=wall, linewidth=20, color=cmap[wall], solid_capstyle='round')


def plot_geometry_patches(xm, rm, lab_f, patches, ax):

    alpha = np.linspace(0, 2*np.pi, 500)

    for wall in range(3):

        xp = rm[wall]*np.cos(alpha)
        yp = rm[wall]*np.sin(alpha)
        if rm[wall] > 0:
            ikeep = ((xm[wall]-rm[wall]-xp)>=0)
        elif rm[wall] <= 0:
            ikeep = ((xm[wall]-rm[wall]-xp)<0)

        # Wall arc
        x = xm[wall]-rm[wall]-xp[ikeep]
        y = yp[ikeep]

        # Proportion of each patch length with respect to total wall length
        lab_f_wall = lab_f[patches==wall]
        fracs = np.round(len(x) * lab_f_wall/sum(lab_f_wall))
        fracs_cul = [sum(fracs[0:i[0]]) for i in enumerate(fracs)]
        fracs = np.append(fracs_cul, sum(fracs)).astype(int)
        fracs[-1] = max(fracs[-1], len(x))

        for patch in range(sum(patches==wall)):
            if patch == sum(patches==wall):
                i1 = -1
            else:
                i1 = fracs[patch+1]
            xpatch = x[fracs[patch]:i1]
            ypatch = y[fracs[patch]:i1]
            if xpatch.size == 0:
                xpatch = [0, 0]
                ypatch = [-1, 1]
            ax.plot(-xpatch, ypatch, label=wall, linewidth=20, solid_capstyle='round')


def bullseye(file_path, file_name, data, color_palette="rocket_r", c_lim=None, color_bar=True,
             save_name="bullseye", save_fig=True, show_fig=False, file_type="pdf"):

    fig, ax = plt.subplots(figsize=(8, 10))

    # Construct color map
    cmap = sns.color_palette(color_palette, as_cmap=True)
    if c_lim:
        norm = mpl.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
    else:
        norm = mpl.colors.Normalize(vmin=min(data), vmax=max(data))

    # Assign color to each segment
    cdata = cmap(norm(data))

    # RV
    width = 0.2
    wedgeprops = {'edgecolor': 'black', 'linewidth': 2, 'width': width}
    patches0, _ = ax.pie(1/4*np.ones(4), wedgeprops=wedgeprops, center=(-1, 0), radius=1, startangle=45,
                        colors=np.vstack([cdata[16:19, :], np.ones(4)]))
    patches1, _ = ax.pie(1/3*np.ones(3), wedgeprops=wedgeprops, center=(-1, 0), radius=1-width, startangle=60,
                        colors=np.vstack([cdata[19:21, :], np.ones(4)]))

    # Hide RV segment that is overlapping the LV
    patches0[-1].set_zorder(0)
    patches0[-1].set_edgecolor('white')
    patches1[-1].set_zorder(0)
    patches1[-1].set_edgecolor('white')

    # LV
    width = 0.28
    wedgeprops = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 2, 'width': width}
    ax.pie(1/6*np.ones(6), wedgeprops=wedgeprops, radius=1, startangle=60, colors=cdata[0:6, :])
    ax.pie(1/6*np.ones(6), wedgeprops=wedgeprops, radius=1-width, startangle=60, colors=cdata[6:12, :])
    ax.pie(1/4*np.ones(4), wedgeprops=wedgeprops, radius=1-2*width, startangle=45, colors=cdata[12:16, :])

    # Set axes
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(-2.25, 1.25), ylim=(-1.25, 1.25))

    if color_bar:
        axl = fig.add_axes([0.3, 0.2, 0.4, 0.04])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                     cax=axl, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Activation timing (ms)', size=14)

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()


def graph(file_path, file_name, G, lit_time, t_myo, t_purk, color_palette="rocket_r", c_lim=None,
          color_bar=True, save_name="graph", save_fig=True, show_fig=False, file_type="pdf"):

    # Get nodel positions
    pos = nx.get_node_attributes(G,'pos')

    # Construct color map
    cmap = sns.color_palette(color_palette, as_cmap=True)
    if c_lim:
        norm = mpl.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
    else:
        norm = mpl.colors.Normalize(vmin=min(lit_time), vmax=max(lit_time))

    # Assign color to each segment
    cdata = cmap(norm(lit_time))

    fig, ax = plt.subplots(figsize=(8,10))

    # Add longitudinal for LV and RV

    for r in [1, 0.8, 0.6]:
        circle = plt.Circle( (-1, 0), r, fill = False, color="0.9", linewidth=2)
        ax.add_artist(circle)

    for r in [1, 0.72, 0.44]:
        circle = plt.Circle( (0, 0), r, edgecolor="0.9", facecolor="1", linewidth=2)
        ax.add_artist(circle)

    # Plot edges
    for e, weight in zip(G.edges, list(nx.get_edge_attributes(G, 'weight').values())):

        if weight == t_myo:
            arc = 0
            color = "0.7"
        elif weight == t_purk:
            # arc = 0.17
            arc = 0
            color = "0.3"

        ax.annotate("",
            xy=pos[e[0]], xycoords='data',
            xytext=pos[e[1]], textcoords='data',
            arrowprops=dict(arrowstyle="-", color=color, linewidth=3,
                            shrinkA=9, shrinkB=9, zorder=3,
                            patchA=None, patchB=None,
                            connectionstyle="arc3,rad=-rrr".replace('rrr',str(arc)
                            ),
                            ),
            )

    # Draw nodes
    nodes_draw = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cdata, node_size=350, edgecolors='k', linewidths=1)
    nodes_draw.set_zorder(4)

    plt.axis('off')

    if color_bar:
        axl = fig.add_axes([0.3, 0.2, 0.4, 0.04])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                     cax=axl, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Activation timing (ms)', size=14)

    # Set axes
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(-2.25, 1.25), ylim=(-1.25, 1.25))

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name + "_" + save_name + "." + file_type), bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


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