import numpy as np
import pandas as pd
from .solvers import initialize_solvers_volumes, set_time_vector
from .heart import set_total_wall_volumes_areas
from .utils import get_outputs
from pathlib import Path


def initialize(model, use_converged):
    """
    Initialize growth simulation
    """

    # Initialize time arrays for growth variable storage
    model.growth.n_g = model.growth.time.size
    model.growth.lab_f_max = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.lab_r_max = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.sig_f_max = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.f_g = np.ones((model.growth.n_g, 3, model.heart.n_patches_tot))
    model.growth.s_l = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.s_r = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.s_l_set = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.s_r_set = np.zeros((model.growth.n_g, model.heart.n_patches_tot))
    model.growth.lab_f = np.zeros((model.growth.n_g, model.solver.n_inc, model.heart.n_patches_tot))
    model.growth.sig_f = np.zeros((model.growth.n_g, model.solver.n_inc, model.heart.n_patches_tot))
    model.growth.v_lv = np.zeros((model.growth.n_g, model.solver.n_inc))
    model.growth.p_lv = np.zeros((model.growth.n_g, model.solver.n_inc))
    model.growth.v_la = np.zeros((model.growth.n_g, model.solver.n_inc))
    model.growth.p_la = np.zeros((model.growth.n_g, model.solver.n_inc))
    model.growth.xm_ed = np.zeros((model.growth.n_g, 3))
    model.growth.xm_es = np.zeros((model.growth.n_g, 3))
    model.growth.rm_ed = np.zeros((model.growth.n_g, 3))
    model.growth.rm_es = np.zeros((model.growth.n_g, 3))      # Cardiac geometry [mm]
    model.growth.outputs = pd.DataFrame()

    # Set initial estimates of the global and local solver
    initialize_solvers_volumes(model, use_converged)


def store(model):
    """
    Store hemodynamics and heart values throughout growth
    """

    # Get outputs
    outputs_frame = get_outputs(model, time_g=model.growth.time[model.growth.i_g])
    model.growth.outputs = pd.concat([model.growth.outputs, outputs_frame])

    # Calculate changes in certain metrics during last time step
    if model.growth.i_g == len(model.growth.time) - 1:
        model.growth.outputs['dLVEDV'] = (model.growth.outputs['LVEDV'] - model.growth.outputs['LVEDV'].loc[-1]) / model.growth.outputs['LVEDV'].loc[-1]
        model.growth.outputs['dLVESV'] = (model.growth.outputs['LVESV'] - model.growth.outputs['LVESV'].loc[-1]) / model.growth.outputs['LVESV'].loc[-1]
        model.growth.outputs['dLVEF'] = (model.growth.outputs['LVEF'] - model.growth.outputs['LVEF'].loc[-1]) / model.growth.outputs['LVEF'].loc[-1]
        model.growth.outputs['dEDWthLfw'] = (model.growth.outputs['EDWthLfw'] - model.growth.outputs['EDWthLfw'].loc[-1]) / model.growth.outputs['EDWthLfw'].loc[-1]
        model.growth.outputs['dESWthLfw'] = (model.growth.outputs['ESWthLfw'] - model.growth.outputs['ESWthLfw'].loc[-1]) / model.growth.outputs['ESWthLfw'].loc[-1]
        model.growth.outputs['dEDWthRfw'] = (model.growth.outputs['EDWthRfw'] - model.growth.outputs['EDWthRfw'].loc[-1]) / model.growth.outputs['EDWthRfw'].loc[-1]
        model.growth.outputs['dESWthRfw'] = (model.growth.outputs['ESWthRfw'] - model.growth.outputs['ESWthRfw'].loc[-1]) / model.growth.outputs['ESWthRfw'].loc[-1]
        model.growth.outputs['dEDWthSw'] = (model.growth.outputs['EDWthSw'] - model.growth.outputs['EDWthSw'].loc[-1]) / model.growth.outputs['EDWthSw'].loc[-1]
        model.growth.outputs['dESWthSw'] = (model.growth.outputs['ESWthSw'] - model.growth.outputs['ESWthSw'].loc[-1]) / model.growth.outputs['ESWthSw'].loc[-1]

    # Store stretch, volume, and pressure
    model.growth.lab_f[model.growth.i_g, :, :] = model.heart.lab_f
    model.growth.sig_f[model.growth.i_g, :, :] = model.heart.sig_f
    model.growth.lab_f_max[model.growth.i_g, :] = model.heart.lab_f.max(axis=0)
    model.growth.lab_r_max[model.growth.i_g, :] = (1/model.heart.lab_f**2).max(axis=0)
    model.growth.sig_f_max[model.growth.i_g, :] = model.heart.sig_f.max(axis=0)
    model.growth.v_lv[model.growth.i_g, :] = model.volumes[:, 2]
    model.growth.p_lv[model.growth.i_g, :] = model.pressures[:, 2]
    model.growth.v_la[model.growth.i_g, :] = model.volumes[:, 1]
    model.growth.p_la[model.growth.i_g, :] = model.pressures[:, 1]

    # Geometry at ES and ED
    model.growth.xm_ed[model.growth.i_g, :] = model.heart.xm[outputs_frame['IED'], :]
    model.growth.xm_es[model.growth.i_g, :] = model.heart.xm[outputs_frame['IES'], :]
    model.growth.rm_ed[model.growth.i_g, :] = model.heart.rm[outputs_frame['IED'], 0:3]
    model.growth.rm_es[model.growth.i_g, :] = model.heart.rm[outputs_frame['IES'], 0:3]


def update_circ_heart(model):
    """
    Update heart and circulation parameters throughout growth
    """

    # Cardiac cycle timing
    model.circulation.hr = model.growth.hr[model.growth.i_g]
    set_time_vector(model)

    # Hemodynamics
    model.circulation.sbv = model.growth.sbv[model.growth.i_g]
    model.resistances.ras = model.growth.ras[model.growth.i_g]
    model.resistances.rmvb = model.growth.rmvb[model.growth.i_g]

    # Activation timing
    model.heart.t_act = model.growth.t_act[model.growth.i_g, :]

    # Ischemia - disable active contraction of ischemic regions
    model.heart.ischemic = model.growth.ischemic[model.growth.i_g,:]
    model.heart.sf_act = model.heart.sf_act * (1 - model.heart.ischemic)
    
    if model.growth.type == "nonmechanic":
        # Age
        x = model.growth.i_g 

        model.circulation.hr = 1.2**(-x+22) + 80

        #Weight depending on age x
        # This was modeled based on Christian's data
        weight = 9.94-0.895*x+0.702*x**2-0.0492*x**3+.00151*x**4-.0000216*x**5+.000000118*x**6
        
        # Calculate Blood Volume:
        # Avg Blood Volume = Patient weight (kg) * avg blood volume
        # Avg blood volume for adult female: 65 mL/kg
        # "" ""  ""        for infants: 80mL/kg
        # "" ""  ""        for neonates: 85mL/kg
        # "" ""  ""        for premature neonates: 95mL/kg

        avg_bv = 20**(-0.05*x + 1) + 65

        # Stressed blood volume is about 30% of total (15-30% range)
        model.circulation.sbv = 0.3*(weight * avg_bv)

        pars_scaled = {}

        # Parameters for a healthy 75kg male
        resistances = {
            "rvp": 0.015,
            "rcs": 0.09,
            "ras": 0.900,
            "rvs": 0.015,
            "rcp": 0.020,
            "rap": 0.300,
            "rav": 0.025
        }
        capacitances = {
            "cvp": 8.0,
            "cas": 1.12,
            "cvs": 70.0,
            "cap": 13.0
        }

        for key,value in resistances.items():
            pars_scaled[key] = hiebing_scaling(value, -3/4, weight)

        for key,value in capacitances.items():
            pars_scaled[key] = hiebing_scaling(value, 1, weight)

        model.change_pars(pars_scaled)
        
    if model.growth.type == "pediatric":
        # Age in months
        x = model.growth.i_g 

        # Use weight data from birth to 36 months from CDC
        path = Path("input_files") / "wtageinf.xls"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Make sure the file is in the input_files folder.")

        try:
            wtage = pd.read_excel(path)  #load file with age and weight data
        except Exception as e:
            try:
                wtage = pd.read_excel(path, engine="xlrd")
            except Exception as e2:
                raise RuntimeError(f"Failed to read {path}: {e}; fallback engine error: {e2}")


        # Weight depending on age x (in months) 50th percentile
        weight = wtage['P50'].iloc[x]

        # Calculate heart rate
        # HR for adult 70kg: 70bpm (Hiebing 2023)
        model.circulation.hr =  hiebing_scaling(70, -0.25, weight)

        # Calculate Blood Volume:
        # SBV for adult 70kg: 880mL (Hiebing 2023)
        model.circulation.sbv = hiebing_scaling(880, 1, weight)
        print("sbv: ", model.circulation.sbv)

        pars_scaled = {}
        # Parameters for a healthy 75kg male
        resistances = {
            "rvp": 0.015,
            "rcs": 0.09,
            "ras": 0.900,
            "rvs": 0.015,
            "rcp": 0.020,
            "rap": 0.300,
            "rav": 0.025
        }
        capacitances = {
            "cvp": 8.0,
            "cas": 1.12,
            "cvs": 70.0,
            "cap": 13.0
        }

        for key,value in resistances.items():
            pars_scaled[key] = hiebing_scaling(value, -3/4, weight)

        for key,value in capacitances.items():
            pars_scaled[key] = hiebing_scaling(value, 1, weight)

        model.change_pars(pars_scaled)

        model.growth.ras[x] = hiebing_scaling(0.900, -3/4, weight)

        print('ras: ', model.growth.ras)

def grow(model):
    """
    Determine growth tensor and update LV geometry
    """

    # Determine time step
    dt = model.growth.time[model.growth.i_g] - model.growth.time[model.growth.i_g - 1]

    # Get growth tensor for all patches at previous time increement
    f_g_old = model.growth.f_g[model.growth.i_g - 1, :, :]

    # Get change in growth tensor
    if model.growth.type == "isotropic_oomen":
        f_g = fg_isotropic(model, f_g_old, dt)
    elif model.growth.type == "transverse_witzenburg":
        f_g = fg_transverse(model, f_g_old, dt)
    elif model.growth.type == "transverse_jones":
        f_g = fg_transverse_goktepe(model, f_g_old, dt)
    elif model.growth.type == "isotropic_jones":
        f_g = fg_isotropic_goktepe(model, f_g_old, dt)
    elif model.growth.type == "transverse_hybrid":
        f_g = fg_hybrid(model, f_g_old, dt)
    elif model.growth.type == "nonmechanic":
        f_g = fg_nonmechanic(model, f_g_old, dt)
    elif model.growth.type == "pediatric":
        f_g = fg_pediatric(model, f_g_old, dt)
    else:
        raise Exception("Growth type not recognized")

    # Update growth tensor, supress growth in non-growing walls
    i_not_growing_walls = [i for i, patch in enumerate(model.heart.patches) if patch not in model.growth.growing_walls]
    f_g[:, i_not_growing_walls] = 1
    f_g_dot = f_g / f_g_old
    # print("this is fg dot", f_g_dot)

    model.growth.f_g[model.growth.i_g, :, :] = f_g

    if model.growth.type == "nonmechanic":
        x = model.growth.i_g 
        #LV Wall Mass
        # Function that fits Christian's data
        lvmass = 20.3 + -2.44*x + 0.963*x**2 + -0.0529*x**3 + 0.00116*x**4 + -0.00000929*x**5 + 0.00000000544*x**6
        print("lv mass", lvmass)

        #LV Wall Volume
        lvwv = lvmass / 1.05 * 1e3
        print("lv volume ", lvwv)

        # Volumetric scaling factor
        theta_lv = lvwv / model.outputs["LVWV"][0]

        # Adjust LV wall volume
        model.change_pars({"LVWV": lvwv})

        # print("lvwv", lvwv)

        # Adjust LV free wall area
        model.change_pars({"AmRefLfw": model.heart.am_ref[0] * theta_lv **(2/3)})

        # Use ratios to adjust other walls
        model.change_pars({"AmRefRfwRatio": 1.36, "AmRefSwRatio": 0.53, "AmRefLARatio": 0.70, "AmRefRARatio": 0.51,
                "RfwVRatio": 0.584, "LAWVRatio": 0.0924, "RAWVRatio": 0.0410})

    else:
        # Update geometry, including total wall volume and midwall reference areas
        model.heart.am_ref = np.maximum(model.heart.am_ref * f_g_dot[0, :] * f_g_dot[1, :], 0.01)
        model.heart.vw = model.heart.vw * np.prod(f_g_dot, axis=0) 
    
    set_total_wall_volumes_areas(model)


def fg_isotropic(model, f_g, dt):
    """
    Determine isotropic growth tensor using evolving fiber strain setpoint based on Oomen 2022
    """

    # Calculate maximum Green-Lagrange strain time history and most recent value (at i_g-1)
    eps_f_max         = 0.5 * (model.growth.lab_f_max[0:model.growth.i_g, :]**2 - 1)
    eps_f_max_current = 0.5 * (model.growth.lab_f_max[model.growth.i_g-1, :]**2 - 1)

    # Get fading memory of maximum stretch
    eps_f_max_set = get_weighted_average(eps_f_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)

    # Stimulus function, where ischemic regions don't grow
    s_f = (eps_f_max_current - eps_f_max_set) * (1 - model.heart.ischemic)

    # If n is odd, s50 is to be subtracted rather than added in the sigmoid in the reversal part (s<0)
    is_even = (model.growth.n_f_min % 2 == False) * 2 - 1

    f_g_dot = np.ones_like(f_g)
    for i_patch in range(len(s_f)):
        if s_f[i_patch] > 0:
            f_g_dot[0:2, i_patch] = np.sqrt(s_f[i_patch] ** model.growth.n_f_plus / (s_f[i_patch] ** model.growth.n_f_plus + model.growth.s50_f_plus ** model.growth.n_f_plus) * dt * model.growth.fgmax_f_plus + 1)
        elif s_f[i_patch] < 0:
            f_g_dot[0:2, i_patch] = np.sqrt(-s_f[i_patch]**model.growth.n_f_min / (s_f[i_patch] ** model.growth.n_f_min + is_even * model.growth.s50_f_min ** model.growth.n_f_min) * dt * model.growth.fgmax_f_min + 1)
        else:
            f_g_dot[0:2, i_patch] = 1.0

    f_g_dot[2, :] = model.growth.fr_ratio*(f_g_dot[0,:] - 1) + 1

    # Store growth stimuli
    model.growth.s_l[model.growth.i_g, :] = s_f

    # Update growth tensor
    return f_g * f_g_dot


def fg_transverse(model, f_g, dt):
    """
    Determine transversely isotropic growth tensor based on Witzenburg & Holmes 2021
    """

    # Calculate maximum Green-Lagrange strain time history
    eps_f_max = 0.5 * (model.growth.lab_f_max[0:model.growth.i_g, :]**2 - 1)
    eps_r_max = 0.5 * (model.growth.lab_r_max[0:model.growth.i_g, :]**2 - 1)

    # Get fading memory of maximum stretch
    eps_f_max_set = get_weighted_average(eps_f_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)
    eps_r_max_set = get_weighted_average(eps_r_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)

    s_f = (eps_f_max[-1, :] - eps_f_max_set[0, :]) * (1 - model.heart.ischemic)
    s_r = (-eps_r_max[-1, :] + eps_r_max_set[0, :]) * (1 - model.heart.ischemic)

    # If n is odd, s50 is to be subtracted rather than added in the sigmoid in the reversal part (s<0)
    is_even = (model.growth.n_f_min % 2 == False) * 2 - 1
    is_even_r = (model.growth.n_r_min % 2 == False) * 2 - 1

    # Change in fiber growth in this iteration
    f_g_dot = np.ones_like(f_g)
    for i_patch in range(len(s_f)):
        if s_f[i_patch] > 0:
            f_g_dot[0:2, i_patch] = np.sqrt(s_f[i_patch] ** model.growth.n_f_plus / (s_f[i_patch] ** model.growth.n_f_plus + model.growth.s50_f_plus ** model.growth.n_f_plus) * dt * model.growth.fgmax_f_plus + 1)
        elif s_f[i_patch] < 0:
            f_g_dot[0:2, i_patch] = np.sqrt(-s_f[i_patch]**model.growth.n_f_min / (s_f[i_patch] ** model.growth.n_f_min + is_even * model.growth.s50_f_min ** model.growth.n_f_min) * dt * model.growth.fgmax_f_plus + 1)
        else:
            f_g_dot[0:2, i_patch] = 1.0

    # Change in radial growth in this iteration
    for i_patch in range(len(s_r)):
        if s_r[i_patch] > 0:
            f_g_dot[2, i_patch] = s_r[i_patch] ** model.growth.n_r_plus / (s_r [i_patch] ** model.growth.n_r_plus + model.growth.s50_r_plus ** model.growth.n_r_plus) * dt * model.growth.fgmax_r_plus + 1
        elif s_r[i_patch] < 0:
            f_g_dot[2, i_patch] = -s_r[i_patch]**model.growth.n_r_min / (s_r [i_patch] ** model.growth.n_r_min + is_even_r * model.growth.s50_r_min ** model.growth.n_r_min) * dt * model.growth.fgmax_r_min + 1
        else:
            f_g_dot[2, i_patch] = 1.0

    # Store growth stimuli and setpoints
    model.growth.s_l[model.growth.i_g, :] = s_f
    model.growth.s_r[model.growth.i_g, :] = s_r
    model.growth.s_l_set[model.growth.i_g, :] = eps_f_max_set
    model.growth.s_r_set[model.growth.i_g, :] = eps_r_max_set

    # Update growth tensor
    return f_g * f_g_dot


def fg_transverse_goktepe(model, f_g_old, dt):
    """
    Determine transversely isotropic growth tensor from Jones & Oomen, 2024
    """

    # Get mechanics
    lab_f_max = model.growth.lab_f_max[0:model.growth.i_g, :]
    lab_r_max = model.growth.lab_r_max[0:model.growth.i_g, :]
    lab_r_max = model.growth.sig_f_max[0:model.growth.i_g, :]

    # Get fading memory of maximum stretch
    lab_f_max_set = get_weighted_average(lab_f_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)
    lab_r_max_set = get_weighted_average(lab_r_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)

    # Get current growth multipliers
    theta_f = f_g_old[0, :] ** 2
    theta_r = f_g_old[2, :]

    # Compute stimulus functions
    s_f = (lab_f_max[-1, :] - lab_f_max_set)/lab_f_max_set * (1 - model.heart.ischemic)
    s_r = (-lab_r_max[-1, :] + lab_r_max_set)/lab_f_max_set * (1 - model.heart.ischemic)
    s_r = (lab_r_max[-1, :] - lab_r_max_set)/lab_r_max_set * (1 - model.heart.ischemic)

    phi_f = s_f / model.growth.tau_f_min * (s_f < 0) + s_f / model.growth.tau_f_plus * (s_f >= 0)
    phi_r = s_r / model.growth.tau_r_min * (s_r < 0) + s_r / model.growth.tau_r_plus * (s_r >= 0)

    # Compute weighing function
    k_f = k_g(theta_f, model.growth.theta_f_min, model.growth.gamma) * (theta_f < 1) + \
          k_g(theta_f, model.growth.theta_f_max, model.growth.gamma) * (theta_f >= 1)
    k_r = k_g(theta_r, model.growth.theta_r_min, model.growth.gamma) * (theta_r < 1) + \
          k_g(theta_r, model.growth.theta_r_max, model.growth.gamma) * (theta_r >= 1)

    # Growth multiplier update
    theta_f = theta_f + k_f * phi_f * dt
    theta_r = theta_r + k_r * phi_r * dt

    # Updated growth tensor
    f_g = np.ones_like(f_g_old)
    f_g[0, :] = f_g[1, :] = np.sqrt(theta_f)
    f_g[2, :] = theta_r

    # Prevent growth beyond min and max values, can occur if time step is too big
    f_g = np.clip(f_g.T, [model.growth.theta_f_min, model.growth.theta_f_min, model.growth.theta_r_min],
                         [model.growth.theta_f_max**2, model.growth.theta_f_max**2, model.growth.theta_r_max]).T

    # Store growth stimuli and setpoints
    model.growth.s_l[model.growth.i_g, :] = s_f
    model.growth.s_r[model.growth.i_g, :] = s_r
    model.growth.s_l_set[model.growth.i_g, :] = lab_f_max_set
    model.growth.s_r_set[model.growth.i_g, :] = lab_r_max_set

    return f_g


def fg_isotropic_goktepe(model, f_g_old, dt):
    """
    Determine isotropic growth tensor based on Jones & Oomen, 2024
    """

    # Get mechanics
    lab_f_max = model.growth.lab_f_max[0:model.growth.i_g, :]

    # Get fading memory of maximum stretch
    lab_f_max_set = get_weighted_average(lab_f_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)

    # Get current growth multipliers
    theta_f = f_g_old[0, :] ** 2

    # Compute stimulus functions
    s_f = (lab_f_max[-1, :] - lab_f_max_set)/lab_f_max_set * (1 - model.heart.ischemic)

    phi_f = s_f / model.growth.tau_f_min * (s_f < 0) + s_f / model.growth.tau_f_plus * (s_f >= 0)

    # Compute weighing function
    k_f = k_g(theta_f, model.growth.theta_f_min, model.growth.gamma) * (theta_f < 1) + \
          k_g(theta_f, model.growth.theta_f_max, model.growth.gamma) * (theta_f >= 1)

    # Growth multiplier update
    theta_f = theta_f + k_f * phi_f * dt

    # Updated growth tensor
    f_g = np.ones_like(f_g_old)
    f_g[0, :] = f_g[1, :] = f_g[2, :] = theta_f**(1.0/3.0)

    # Prevent growth beyond min and max values, can occur if time step is too big
    f_g = np.clip(f_g.T, [model.growth.theta_f_min, model.growth.theta_f_min, model.growth.theta_r_min],
                         [model.growth.theta_f_max**2, model.growth.theta_f_max**2, model.growth.theta_r_max]).T

    # Store growth stimuli and setpoints
    model.growth.s_l[model.growth.i_g, :] = s_f
    model.growth.s_l_set[model.growth.i_g, :] = lab_f_max_set

    return f_g

def fg_nonmechanic(model, f_g_old, dt):
    """ 
    Simple growth model that does not rely on mechanics
    """

    # Get current growth multipliers
    theta_f = f_g_old[0, :] 

    # # We want to have a linear growth up until age 15
    # if model.growth.i_g <= 15:
    #     phi = 0.05
    # else:
    #     phi = 1/(10*(model.growth.i_g))

    phi = 1/(4*model.growth.i_g)
    # # Growth multiplier update
    theta_f = theta_f + model.growth.tau_f_plus* dt * (phi) 

    # Updated growth tensor
    f_g = np.ones_like(f_g_old)
    f_g[0, :] = f_g[1, :] = f_g[2, :] = theta_f

    return f_g

def fg_hybrid(model, f_g_old, dt):
    """Hybrid growth law between the Witzenburg and Jones & Oomen laws: sigmoid growth rate function with growth limiter
    based on total growth"""

    # Get mechanics
    lab_f_max = model.growth.lab_f_max[0:model.growth.i_g, :]
    lab_r_max = model.growth.lab_r_max[0:model.growth.i_g, :]

    # Get fading memory of maximum stretch
    lab_f_max_set = get_weighted_average(lab_f_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)
    lab_r_max_set = get_weighted_average(lab_r_max, model.growth.t_mem, model.growth.time, model.growth.i_g - 1)

    # Get current growth multipliers
    theta_f = f_g_old[0, :] ** 2
    theta_r = f_g_old[2, :]

    # Compute stimulus functions
    s_f = (lab_f_max[-1, :] - lab_f_max_set) * (1 - model.heart.ischemic)
    s_r = (-lab_r_max[-1, :] + lab_r_max[0, :]) * (1 - model.heart.ischemic)

    phi_f = -hill_function(s_f, model.growth.s50_f_min, model.growth.n_f_min, model.growth.tau_f_min) * (s_f < 0) + \
             hill_function(s_f, model.growth.s50_f_plus, model.growth.n_f_plus, model.growth.tau_f_plus) * (s_f >= 0)
    phi_r = -hill_function(s_r, model.growth.s50_r_min, model.growth.n_r_min, model.growth.tau_r_min) * (s_r < 0) + \
             hill_function(s_r, model.growth.s50_r_plus, model.growth.n_r_plus, model.growth.tau_r_plus) * (s_r >= 0)

    # Compute weighing function
    k_f = k_g(theta_f, model.growth.theta_f_min, model.growth.gamma) * (theta_f < 1) + \
          k_g(theta_f, model.growth.theta_f_max, model.growth.gamma) * (theta_f >= 1)
    k_r = k_g(theta_r, model.growth.theta_r_min, model.growth.gamma) * (theta_r < 1) + \
          k_g(theta_r, model.growth.theta_r_max, model.growth.gamma) * (theta_r >= 1)

    # Growth multiplier update
    theta_f = theta_f + k_f * phi_f * dt
    theta_r = theta_r + k_r * phi_r * dt

    # Updated growth tensor
    f_g = np.ones_like(f_g_old)
    f_g[0, :] = f_g[1, :] = np.sqrt(theta_f)
    f_g[2, :] = theta_r

    # Prevent growth beyond min and max values, can occur if time step is too big
    f_g = np.clip(f_g.T, [model.growth.theta_f_min, model.growth.theta_f_min, model.growth.theta_r_min],
                         [model.growth.theta_f_max, model.growth.theta_f_max, model.growth.theta_r_max]).T

    # Store growth stimuli
    model.growth.s_l[model.growth.i_g, :] = s_f
    model.growth.s_r[model.growth.i_g, :] = s_r
    model.growth.s_l_set[model.growth.i_g, :] = lab_f_max_set
    model.growth.s_r_set[model.growth.i_g, :] = lab_r_max_set

    return f_g

def fg_pediatric(model, f_g_old, dt):
    """
    Based off Hiebing 2023
    """

    # f_g is just 1
    f_g = np.ones_like(f_g_old)

    return f_g



def get_weighted_average(y, window_time, time, i_previous):
    """Calculate weighted average using a sawtooth weighting function"""

    # Round window time to nearest integer
    window_time = round(window_time)

    # Make sawtooth shaped weighting function
    weights = np.concatenate((np.arange(1, np.ceil(window_time / 2) + 1), np.arange(np.floor(window_time / 2), 0, -1)))
    weights_time = np.arange(time[i_previous] - np.ceil(window_time), time[i_previous])

    # Clip weights time to -1, this will repeat the first time point to fill up the history window is larger than the
    # number of time points that have passed since the start of the simulation
    weights_time = np.clip(weights_time, -1, None)

    # Find indices of weights time in current time history
    i_y = np.searchsorted(time[:i_previous], weights_time)

    # Calculate weighted average of maximum stretch
    return np.average(y[i_y, :], weights=weights, axis=0)


def hill_function(s, s_50, n, tau):
    """Hill function"""
    return abs(s)**n / (abs(s)**n + abs(s_50)**n) / tau


def k_g(theta, theta_max, gamma):
    """Weighting function"""
    return  ((theta_max - theta) / (theta_max - 1))**gamma


# Parameter scaling formula from Hiebing 2023
def hiebing_scaling(y, b, w, w0=70):
    return y * (w/w0)**b