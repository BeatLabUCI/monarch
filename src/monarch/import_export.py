import os
import numpy as np
import json
import h5py
import pathlib
import pandas as pd
from dataclasses import dataclass
from .heart import get_wall_thickness


def import_pars(model, model_pars):
    """Import model parameters from input file and assign to model class"""
    # Load last converged solution
    with open(os.path.join(str(model_pars) + ".json")) as json_file:
        pars = json.load(json_file)

    # Assign self parameters to data classes

    @dataclass
    class Circulation:
        hr = pars['Circulation']['hr']  # [1/s]
        sbv = pars['Circulation']['sbv']  # [mL]
        k = np.array(pars['Circulation']['k'])  # [-]
        # if bsa is specified
        if 'bsa' in pars['Circulation']:
            bsa = pars['Circulation']['bsa']  # [m^2]
        else:
            bsa = None

    @dataclass
    class Capacitances:  # [mL/mmHg]
        cvp = pars['Capacitances']['cvp']
        cas = pars['Capacitances']['cas']
        cvs = pars['Capacitances']['cvs']
        cap = pars['Capacitances']['cap']

    @dataclass
    class Resistances:  # [mmHg * s/mL]
        rvp = pars['Resistances']['rvp']
        rcs = pars['Resistances']['rcs']
        ras = pars['Resistances']['ras']
        rvs = pars['Resistances']['rvs']
        rcp = pars['Resistances']['rcp']
        rap = pars['Resistances']['rap']
        rav = pars['Resistances']['rav']
        rmvb = pars['Resistances']['rmvb']
        rtvb = pars['Resistances']['rtvb']
        if rmvb > 1e10:
            rmvb = np.inf
        if rtvb > 1e10:
            rtvb = np.inf

    @dataclass
    class Solver:
        cutoff = pars['Solver']['cutoff']
        iter_max = pars['Solver']['iter_max']
        n_inc = pars['Solver']['n_inc']

    @dataclass
    class Heart:
        # Heart geometry
        patches = np.array(pars['Heart']['patches'])
        am_ref = np.array(pars['Heart']['am_ref'])
        vw = np.array(pars['Heart']['wv'])

        # Active and passive Material properties
        ls_ref = np.array(pars['Heart']['ls_ref'])  # [um]
        ls_eiso = np.array(pars['Heart']['ls_eiso'])  # [um]
        lsc_0 = np.array(pars['Heart']['lsc0'])  # [um]
        v_max = np.array(pars['Heart']['v_max'])  # [um/ms]
        tau_r = np.array(pars['Heart']['tr'])  # [-]
        tau_d = np.array(pars['Heart']['td'])  # [-]
        t_ad = np.array(pars['Heart']['tad'])  # [ms]
        sf_act = (np.array(pars['Heart']['sf_act']) * (
                1 - np.array(pars['Heart']['ischemic'])))  # [MPa], accounting for ischemia
        c_1 = np.array(pars['Heart']['c_1'])  # [MPa]
        c_3 = np.array(pars['Heart']['c_3'])  # [MPa]
        c_4 = np.array(pars['Heart']['c_4'])  # [-]
        t_act = np.array(pars['Heart']['t_act'])  # [ms]

    @dataclass
    class Pericardium:
        c_1 = np.array(pars['Pericardium']['c_1'])  # [MPa]
        c_3 = np.array(pars['Pericardium']['c_3'])  # [MPa]
        c_4 = np.array(pars['Pericardium']['c_4'])  # [-]
        thickness = np.array(pars['Pericardium']['thickness'])  # [mm]
        lab_pre = np.array(pars['Pericardium']['pre_stretch'])  # [-]

    @dataclass
    class Growth:
        time = np.array(pars['Growth']['time'])
        hr = np.array(pars['Growth']['hr'])
        rmvb = np.array(pars['Growth']['rmvb'])
        sbv = np.array(pars['Growth']['sbv'])
        ras = np.array(pars['Growth']['ras'])
        n_g = time.size

        n_patches = np.array(pars['Growth']['ischemic'][0]['ischemic']).size
        ischemic = np.zeros((len(time), n_patches))
        for ischemic_i in pars['Growth']['ischemic']:
            ischemic[ischemic_i['t_init']:, :] = np.array(ischemic_i['ischemic'])

        t_act = np.zeros((len(time), n_patches))
        for t_i in pars['Growth']['t_act']:
            t_act[t_i['t_init']:, :] = np.array(t_i['t_act'])

        t_mem = pars["Growth"]["t_mem"]
        growing_walls = pars["Growth"]["growing_walls"]

        # Growth type
        type = pars["Growth"]["type"]

        # Sigmoid growth parameters - fiber direction
        if "fgmax_f_plus" in pars["Growth"]:
            fgmax_f_plus = pars["Growth"]["fgmax_f_plus"]
        if "fgmax_f_min" in pars["Growth"]:
            fgmax_f_min = pars["Growth"]["fgmax_f_min"]
        if "n_f_plus" in pars["Growth"]:
            n_f_plus = round(pars["Growth"]["n_f_plus"])
        if "n_f_min" in pars["Growth"]:
            n_f_min = round(pars["Growth"]["n_f_min"])
        if "s50_f_plus" in pars["Growth"]:
            s50_f_plus = pars["Growth"]["s50_f_plus"]
        if "s50_f_min" in pars["Growth"]:
            s50_f_min = pars["Growth"]["s50_f_min"]

        # Sigmoid growth parameters - radial direction, only if type is not "isotropic"
        if type != "isotropic":
            if "fgmax_r_plus" in pars["Growth"]:
                fgmax_r_plus = pars["Growth"]["fgmax_r_plus"]
            if "fgmax_r_min" in pars["Growth"]:
                fgmax_r_min = pars["Growth"]["fgmax_r_min"]
            if "n_r_plus" in pars["Growth"]:
                n_r_plus = round(pars["Growth"]["n_r_plus"])
            if "n_r_min" in pars["Growth"]:
                n_r_min = round(pars["Growth"]["n_r_min"])
            if "s50_r_plus" in pars["Growth"]:
                s50_r_plus = pars["Growth"]["s50_r_plus"]
            if "s50_r_min" in pars["Growth"]:
                s50_r_min = pars["Growth"]["s50_r_min"]

        if "fr_ratio" in pars["Growth"]:
            fr_ratio = pars["Growth"]["fr_ratio"]

        # Kuhl-type Growth parameters
        if "tau_f-" in pars["Growth"]:
            tau_f_min = pars["Growth"]["tau_f-"]
        if "tau_f+" in pars["Growth"]:
            tau_f_plus = pars["Growth"]["tau_f+"]
        if "tau_r-" in pars["Growth"]:
            tau_r_min = pars["Growth"]["tau_r-"]
        if "tau_r+" in pars["Growth"]:
            tau_r_plus = pars["Growth"]["tau_r+"]
        if "theta_f_min" in pars["Growth"]:
            theta_f_min = pars["Growth"]["theta_f_min"]
        if "theta_f_max" in pars["Growth"]:
            theta_f_max = pars["Growth"]["theta_f_max"]
        if "theta_r_min" in pars["Growth"]:
            theta_r_min = pars["Growth"]["theta_r_min"]
        if "theta_r_max" in pars["Growth"]:
            theta_r_max = pars["Growth"]["theta_r_max"]
        if "gamma" in pars["Growth"]:
            gamma = pars["Growth"]["gamma"]

    # Assign input classes to the self class
    model.circulation = Circulation()
    model.capacitances = Capacitances()
    model.resistances = Resistances()
    model.solver = Solver()
    model.heart = Heart()
    model.pericardium = Pericardium()
    model.growth = Growth()


def store_converged_sol(model):
    """
    Store converged solution, do some checks first to not store awkward solutions
    """

    if not (np.isnan(model.volumes).any()) and (np.isreal(model.volumes).any()) and ((model.volumes[0, :] >= 0).any()):
        k = model.volumes[0, :] / sum(model.volumes[0, :])
        converged_sol = {
            'vs':   model.heart.vs,
            'ys':   model.heart.ys,
            'dv':   model.heart.dv,
            'dy':   model.heart.dy,
            'c':    model.heart.c.tolist(),
            'lsc':  model.heart.lsc.tolist(),
            'k':    k.tolist()
        }
        with open(model.converged_file, 'w') as json_file:
            json.dump(converged_sol, json_file)


def load_converged_sol(model):
    """
    Load last converged solution
    """

    with open(model.converged_file) as json_file:
        converged_sol = json.load(json_file)
        model.heart.c = np.array(converged_sol['c'])
        model.heart.lsc = np.array(converged_sol['lsc'])
        model.circulation.k = np.array(converged_sol['k'])
        model.heart.vs = converged_sol['vs']
        model.heart.ys = converged_sol['ys']
        model.heart.dv = converged_sol['dv']
        model.heart.dy = converged_sol['dy']


def export_beat_sim(model, filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Calculate wall thickness
    wall_thickness, _ = get_wall_thickness(model)

    filepath, filename = check_file_specs(filepath, filename)

    with h5py.File(filepath / filename, "w") as f:
        f.attrs["input_file"] = model.model_pars
        f.create_dataset("time", data=model.time)
        f.create_dataset("pressures", data=model.pressures)
        f.create_dataset("volumes", data=model.volumes)
        f.create_dataset("r_m", data=model.heart.rm)
        f.create_dataset("x_m", data=model.heart.xm)
        f.create_dataset("y_s", data=model.heart.ys_store)
        f.create_dataset("lab_f", data=model.heart.lab_f)
        f.create_dataset("wall_thickness", data=wall_thickness)
        f.create_dataset("t_act", data=model.heart.t_act)
        f.create_dataset("outputs", data=model.outputs.to_numpy())
        f.attrs["outputs_names"] = model.outputs.columns.tolist()
        f.attrs["patches"] = model.heart.patches


def import_beat_sim(model, filepath, filename):
    """Import results from a previous simulation and reload class"""

    filepath, filename = check_file_specs(filepath, filename)

    with h5py.File(filepath / str(filename), "r") as f:
        # Reload class with input file
        model.__init__(f.attrs["input_file"])

        # Load results
        model.time = f["time"][:]
        model.pressures = f["pressures"][:]
        model.volumes = f["volumes"][:]
        model.heart.rm = f["r_m"][:]
        model.heart.xm = f["x_m"][:]
        model.heart.ys_store = f["y_s"][:]
        model.heart.lab_f = f["lab_f"][:]
        model.heart.t_act = f["t_act"][:]
        model.heart.patches = f.attrs["patches"]

        # Create pandas dataframe for outputs
        model.outputs = pd.DataFrame(f["outputs"][:], columns=f.attrs["outputs_names"])


def export_growth_sim(model, filepath, filename):

    check_file_specs(filepath, filename)

    # Create file path if not existing
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with h5py.File(os.path.join(filepath, filename + ".hdf5"), "w") as f:
        f.create_dataset("v_lv", data=model.growth.v_lv)
        f.create_dataset("p_lv", data=model.growth.p_lv)
        f.create_dataset("f_g", data=model.growth.f_g)
        f.create_dataset("s_l", data=model.growth.s_l)
        f.create_dataset("s_r", data=model.growth.s_r)
        f.create_dataset("s_l_set", data=model.growth.s_l_set)
        f.create_dataset("s_r_set", data=model.growth.s_r_set)
        f.create_dataset("lab_f_max", data=model.growth.lab_f_max)
        f.create_dataset("sig_f_max", data=model.growth.sig_f_max)
        f.create_dataset("t_act", data=model.growth.t_act)
        f.create_dataset("outputs", data=model.growth.outputs.to_numpy())
        f.attrs["outputs_columns"] = model.growth.outputs.columns.tolist()
        f.attrs["outputs_rows"] = model.growth.outputs.index.tolist()
        f.attrs["patches"] = model.heart.patches


def check_file_specs(filepath, filename):
    """Check and convert file specifications"""

    # Add .hdf5 if not in filename
    if not filename.endswith(".hdf5"):
        filename = str(filename + ".hdf5")

    # Convert filepath to pathlib.Path if not already
    if not isinstance(filepath, pathlib.Path):
        filepath = pathlib.Path(filepath)

    return filepath, filename
