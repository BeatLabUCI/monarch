import numpy as np
import pandas as pd
from .heart import get_wall_thickness, set_total_wall_volumes_areas, unloaded_heart_volume
from pathlib import Path
import os


def clear_converged_sol():
    """Clear last converged solution"""

    root_dir = Path(__file__).parent.parent.absolute()
    f_converged_sol = Path(os.path.join(root_dir, "io", 'converged_sol.json'))
    if f_converged_sol.is_file():
        os.remove(f_converged_sol)


def get_outputs(model, time_g=0):
    """Collect model outputs in Pandas dataframe"""

    # Get valve opening and closing events and ventricle timing events (isovolumic contraction etc.)
    time_events = get_valve_events(model)

    # Get wall thicknesses and midwall areas
    wall_thickness, a_m_w = get_wall_thickness(model)

    # LV ES and ED
    es_frame = time_events['av_closes']
    ed_frame = time_events['mv_closes']
    es_time = model.time[es_frame]
    esp = model.pressures[es_frame, 2]  # [mmHg]
    esv = model.volumes[es_frame, 2]  # [mL]
    ed_time = model.time[ed_frame]
    edp = model.pressures[ed_frame, 2]  # [mmHg]
    edv = model.volumes[ed_frame, 2]  # [mL]

    # RV ES and ED
    es_frame_rv = time_events['rv_closes']
    ed_frame_rv = time_events['tv_closes']
    es_time_rv = model.time[es_frame_rv]
    esp_rv = model.pressures[es_frame_rv, 6]  # [mmHg]
    esv_rv = model.volumes[es_frame_rv, 6]  # [mL]
    ed_time_rv = model.time[ed_frame_rv]
    edp_rv = model.pressures[ed_frame_rv, 6]  # [mmHg]
    edv_rv = model.volumes[ed_frame_rv, 6]  # [mL]

    # Maximum pressures and gradients for LV and RV
    p_max = max(model.pressures[:, 2])      # [mmHg]
    dpdt_max = max(np.gradient(model.pressures[:, 2], model.time[1] - model.time[0]))  # [mmHg/s]
    dpdt_min = min(np.gradient(model.pressures[:, 2], model.time[1] - model.time[0]))  # [mmHg/s]
    p_max_rv = max(model.pressures[:, 6])      # [mmHg]
    dpdt_max_rv = max(np.gradient(model.pressures[:, 6], model.time[1] - model.time[0]))  # [mmHg/s]
    dpdt_min_rv = min(np.gradient(model.pressures[:, 6], model.time[1] - model.time[0]))  # [mmHg/s]

    # ED wall thickness [mm]
    ed_wth = wall_thickness[ed_frame]
    es_wth = wall_thickness[es_frame]

    # LV and RV Wall Volume
    lvwv = np.sum(model.heart.vw[model.heart.patches == 0]) + np.sum(model.heart.vw[model.heart.patches == 2])
    rvwv = np.sum(model.heart.vw[model.heart.patches == 1])

    # ED wall stretch
    lab_ed_lfw = np.mean(model.heart.lab_f[ed_frame, model.heart.patches == 0])
    lab_ed_rfw = np.mean(model.heart.lab_f[ed_frame, model.heart.patches == 1])
    lab_ed_sw = np.mean(model.heart.lab_f[ed_frame, model.heart.patches == 2])
    lab_ed_la = np.mean(model.heart.lab_f[ed_frame, model.heart.patches == 3])
    lab_ed_ra = np.mean(model.heart.lab_f[ed_frame, model.heart.patches == 4])

    # Cardiac function of LV and RV
    sv = edv - esv  # [mL]
    sv_rv = edv_rv - esv_rv  # [mL]
    ef = (edv - esv) / edv  # [-]
    ef_rv = (edv_rv - esv_rv) / edv_rv  # [-]
    co = sv / model.time[-1] * 60 / 1e3  # [L/min]
    co_rv = sv_rv / model.time[-1] * 60 / 1e3  # [L/min]
    hr = 60 / model.time[-1]  # [s]

    # Calculate regurgitation fraction
    rf, rf_rv = get_rf(model)

    # Arterial function
    map = model.pressures[:, 3].mean()
    sbp = max(model.pressures[:, 3])
    dbp = min(model.pressures[:, 3])

    # Get geometry: LV-SW endocardial distance, endocardial RV-SW distance, and distance between RV insertions
    # Needs to be improved: now half of each wall thickness is substracted but this is not accurate at midwall radius
    dlv_sw = model.heart.xm[ed_frame, 2] - model.heart.xm[ed_frame, 0] - 0.5*(ed_wth[0] + ed_wth[2])
    drv_sw = model.heart.xm[ed_frame, 1] - model.heart.xm[ed_frame, 2] - 0.5*(ed_wth[2] + ed_wth[1])
    drvi = model.heart.ys_store[ed_frame]

    # ED and ES diameter for LV and RV
    lvedd = 2*(abs(model.heart.rm[ed_frame, 0]) - ed_wth[0])
    rvedd = 2*(abs(model.heart.rm[ed_frame_rv, 1]) - ed_wth[1])
    lvesd = 2*(abs(model.heart.rm[es_frame, 0]) - es_wth[0])
    rvesd = 2*(abs(model.heart.rm[es_frame_rv, 0]) - es_wth[1])
    lvfs = (lvedd - lvesd) / lvedd * 100
    rvfs = (rvedd - rvesd) / rvedd * 100

    # mean left atrial pressure
    lap = model.pressures[:, 1].mean()

    # indexed volumes
    if model.circulation.bsa is not None:
        edv_i = edv / model.circulation.bsa
        esv_i = esv / model.circulation.bsa
        rvedv_i = edv_rv / model.circulation.bsa
        rvesv_i = esv_rv / model.circulation.bsa
    else:
        edv_i = np.nan
        esv_i = np.nan
        rvedv_i = np.nan
        rvesv_i = np.nan

    # Turn into pandas
    outputs = pd.DataFrame([[edv, esv, edp, esp, p_max, dpdt_max, dpdt_min, sv, rf, ef, co,
                          edv_rv, esv_rv, edp_rv, esp_rv, p_max_rv, dpdt_max_rv, dpdt_min_rv, sv_rv, rf_rv, ef_rv, co_rv,
                          ed_wth[0], ed_wth[1], ed_wth[2], es_wth[0], es_wth[1], es_wth[2],
                          dlv_sw, drv_sw, drvi, dlv_sw/drv_sw, dlv_sw/drvi, drv_sw/drvi, lvwv, rvwv,
                          lvedd, lvesd, lvfs, rvedd, rvesd, rvfs,
                          lab_ed_lfw, lab_ed_rfw, lab_ed_sw, lab_ed_la, lab_ed_ra,
                          time_events['LVIVCT'], time_events['LVIVRT'], time_events['LVET'], time_events['LVFT'],
                          time_events['RVIVCT'], time_events['RVIVRT'], time_events['RVET'], time_events['RVFT'],
                          map, hr, dbp, sbp, ed_frame, es_frame, ed_time, es_time, ed_frame_rv, es_frame_rv, ed_time_rv,
                          es_time_rv, lap,
                          edv_i, esv_i, rvedv_i, rvesv_i], ],
                        columns=['LVEDV', 'LVESV', 'LVEDP', 'LVESP', 'LVMaxP', 'LVMaxdP', 'LVMindP', 'LVSV', 'LVRF', 'LVEF', 'LVCO',
                                 'RVEDV', 'RVESV', 'RVEDP', 'RVESP', 'RVMaxP', 'RVMaxdP', 'RVMindP', 'RVSV', 'RVRF', 'RVEF', 'RVCO',
                                 'EDWthLfw', 'EDWthRfw', 'EDWthSw', 'ESWthLfw', 'ESWthRfw', 'ESWthSw',
                                 'Dlvsw', 'Drvsw', 'Drvi', 'DlvswDrvsw', 'DlvswDrvi', 'DrvswDrvi', 'LVWV', 'RVWV',
                                 'LVEDD', 'LVESD', 'LVFS', 'RVEDD', 'RVESD', 'RVFS',
                                 'EDStretchLfw', 'EDStretchRfw', 'EDStretchSw', 'EDStretchLA', 'EDStretchRA',
                                 'LVIVCT', 'LVIVRT', 'LVET', 'LVFT',
                                 'RVIVCT', 'RVIVRT', 'RVET', 'RVFT',
                                 'MAP', 'HR', 'DBP', 'SBP',
                                 'IED', 'IES', 'TED', 'TES', 'IED_RV', 'IES_RV', 'TED_RV', 'TES_RV', 'LAP',
                                 'LVEDVi', 'LVESVi', 'RVEDVi', 'RVESVi'],
                        index=[time_g])

    return outputs


def get_valve_events(model):
    """Determine when valves are open (0) and closed (1) based on transvalvular pressure and calculate ventricle
    timings"""
    p = model.pressures
    time_events = {
        "mv_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 2] > p[:, 1], 1)), 0)),
        "mv_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 2] < p[:, 1], 1)), 0)),
        "av_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 3] > p[:, 2], 1)), 0)),
        "av_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 3] < p[:, 2], 1)), 0)),
        "tv_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 6] > p[:, 5], 1)), 0)),
        "tv_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 6] < p[:, 5], 1)), 0)),
        "rv_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 7] > p[:, 6], 1)), 0)),
        "rv_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 7] < p[:, 6], 1)), 0))
    }

    # Isovolumic contraction
    if time_events['av_opens'] > time_events['mv_closes']:
        time_events['LVIVCT'] = (model.time[time_events['av_opens']] - model.time[time_events['mv_closes']])*1e3
    else:
        time_events['LVIVCT'] = (model.time[-1] - model.time[time_events['mv_closes']] + model.time[time_events['av_opens']])*1e3
    if time_events['rv_opens'] > time_events['tv_closes']:
        time_events['RVIVCT'] = (model.time[time_events['rv_opens']] - model.time[time_events['tv_closes']]) * 1e3
    else:
        time_events['RVIVCT'] = (model.time[-1] - model.time[time_events['rv_closes']] + model.time[time_events['rv_opens']]) * 1e3

    # Isovolumic relaxation
    if time_events['av_closes'] < time_events['mv_opens']:
        time_events['LVIVRT'] = (model.time[time_events['mv_opens']] - model.time[time_events['av_closes']])*1e3
    else:
        time_events['LVIVRT'] = (model.time[-1] - model.time[time_events['av_closes']] + model.time[time_events['mv_opens']])*1e3
    if time_events['rv_closes'] < time_events['tv_opens']:
        time_events['RVIVRT'] = (model.time[time_events['tv_opens']] - model.time[time_events['rv_closes']])*1e3
    else:
        time_events['RVIVRT'] = (model.time[-1] - model.time[time_events['rv_closes']] + model.time[time_events['tv_opens']])*1e3

    # Ejection time
    if time_events['av_closes'] > time_events['av_opens']:
        time_events['LVET'] = (model.time[time_events['av_closes']] - model.time[time_events['av_opens']])*1e3
    else:
        time_events['LVET'] = (model.time[-1] - model.time[time_events['av_opens']] + model.time[time_events['av_closes']])*1e3
    if time_events['av_closes'] > time_events['av_opens']:
        time_events['RVET'] = (model.time[time_events['rv_closes']] - model.time[time_events['rv_opens']])*1e3
    else:
        time_events['RVET'] = (model.time[-1] - model.time[time_events['rv_opens']] + model.time[time_events['rv_closes']])*1e3

    # Filling time
    if time_events['mv_closes'] > time_events['mv_opens']:
        time_events['LVFT'] = (model.time[time_events['mv_closes']] - model.time[time_events['mv_opens']])*1e3
    else:
        time_events['LVFT'] = (model.time[-1] - model.time[time_events['mv_opens']] + model.time[time_events['mv_closes']])*1e3
    if time_events['tv_opens'] > time_events['tv_closes']:
        time_events['RVFT'] = (model.time[time_events['tv_closes']] - model.time[time_events['tv_opens']])*1e3
    else:
        time_events['RVFT'] = (model.time[-1] - model.time[time_events['tv_opens']] + model.time[time_events['tv_closes']])*1e3

    return time_events


def get_rf(model):
    """Comute regurgitant fraction of the LV and RV, based on flow equations from beat_it"""

    # Regurgitant volume: integrate flow rate from LV to LA
    dvr = ((model.pressures[:, 2] - model.pressures[:, 1]) /
           model.resistances.rmvb * (model.pressures[:, 2] > model.pressures[:, 1]))
    v_regurgitant = np.trapz(dvr, dx=model.solver.dt)

    # Forward volume: integrate flow rate from LV to systemic arteries
    dvf = (model.pressures[:, 2] - model.pressures[:, 3]) / model.resistances.rcs * (model.pressures[:, 2] > model.pressures[:, 3])
    v_forward = np.trapz(dvf, dx=model.solver.dt)

    # Compute regurgitant fraction, prevent dividing by zero if the heart is really really broken
    if v_forward == 0:
        rf = 0
    else:
        rf = v_regurgitant / (v_forward + v_regurgitant)

    # Regurgitant volume: integrate flow rate from RV to RA
    dvr = ((model.pressures[:, 6] - model.pressures[:, 5]) /
           model.resistances.rtvb * (model.pressures[:, 6] > model.pressures[:, 5]))
    v_regurgitant = np.trapz(dvr, dx=model.solver.dt)

    # Forward volume: integrate flow rate from RV to pulmonary arteries
    dvf = (model.pressures[:, 6] - model.pressures[:, 7]) / model.resistances.rcp * (
                model.pressures[:, 6] > model.pressures[:, 7])
    v_forward = np.trapz(dvf, dx=model.solver.dt)

    # Compute regurgitant fraction, prevent dividing by zero if the heart is really really broken
    if v_forward == 0:
        rf_rv = 0
    else:
        rf_rv = v_regurgitant / (v_forward + v_regurgitant)

    return rf, rf_rv


def change_pars(model, pars):
    """Change model parameter values using a single dictionary input with par_name: par_value. Convenient for changing
    model parameters while fitting. Order is important: ratio-based parameter changes should occur after any absolute
    value changes"""

    # Change all key names to lowercase to prevent case inconsistencies
    pars = {key.lower(): value for key, value in pars.items()}

    # Circulation
    for par, value in pars.items():
        if par == "sbv":
            model.circulation.sbv = value
        elif par == "hr":
            model.circulation.hr = value
        elif par == "k_initial":
            model.circulation.k = value

        # Capacitances
        elif par == "cvp":
            model.capacitances.cvp = value
        elif par == "cas":
            model.capacitances.cas = value
        elif par == "cap":
            model.capacitances.cap = value
        elif par == "cvs":
            model.capacitances.cvs = value

        # Resistances
        elif par == "rvp":
            model.resistances.rvp = value
        elif par == "rcs":
            model.resistances.rcs = value
        elif par == "ras":
            model.resistances.ras = value
        elif par == "rvs":
            model.resistances.rvs = value
        elif par == "rcp":
            model.resistances.rcp = value
        elif par == "rap":
            model.resistances.rap = value
        elif par == "rav":
            model.resistances.rav = value
        elif par == "rmvb":
            model.resistances.rmvb = value
        elif par == "rtvb":
            model.resistances.rtvb = value

        # Heart parameters, ventricles and atria (with suffix _a) separately
        chamber_names = ["", "a"]
        i_ventricles = model.heart.patches < 3
        i_atria = model.heart.patches >= 3
        i_chambers = [i_ventricles, i_atria]
        for i, chambers in enumerate(i_chambers):
            if par == "sact" + chamber_names[i]:
                model.heart.sf_act[chambers] = value
            elif par == "sfact" + chamber_names[i]:
                model.heart.sf_act[chambers] = value
            elif par == "tad" + chamber_names[i]:
                model.heart.t_ad[chambers] = value
            elif par == "td" + chamber_names[i]:
                model.heart.tau_d[chambers] = value
            elif par == "tr" + chamber_names[i]:
                model.heart.tau_r[chambers] = value
            elif par == "c1" + chamber_names[i]:
                model.heart.c_1[chambers] = value
            elif par == "c3" + chamber_names[i]:
                model.heart.c_3[chambers] = value
            elif par == "c4" + chamber_names[i]:
                model.heart.c_4[chambers] = value
            elif par == "tact" + chamber_names[i]:
                model.heart.t_act = value

        # Set parameter for specific patches
        if "_s" in par:
            # Par name and patch number
            patch = int(par.split("_s")[1])
            par = par.split("_")[0]
            if par == "sact":
                model.heart.sf_act[patch] = value
            elif par == "tad":
                model.heart.t_ad[patch] = value
            elif par == "td":
                model.heart.tau_d[patch] = value
            elif par == "tr":
                model.heart.tau_r[patch] = value
            elif par == "c1":
                model.heart.c_1[patch] = value
            elif par == "c3":
                model.heart.c_3[patch] = value
            elif par == "c4":
                model.heart.c_4[patch] = value
            elif par == "tact":
                model.heart.t_act[patch] = value
            elif par == "amref":
                model.heart.am_ref[patch] = value
            elif par == "vw":
                model.heart.vw[patch] = value

        # Timing properties, atriaventricular delay and intraventricular delay (between lfw/rfw and septum)
        if par == "avd":
            model.heart.t_act[i_ventricles] = model.heart.t_act[i_ventricles] + value
        if par == "ivd_lv":
            model.heart.t_act[model.heart.patches == 0] = np.mean(model.heart.t_act[model.heart.patches == 2]) + value
        if par == "ivd_rv":
            model.heart.t_act[model.heart.patches == 1] = np.mean(model.heart.t_act[model.heart.patches == 2]) + value

        # Pericardium
        if par == "wth_p":
            model.pericardium.thickness = value
        elif par == "c1_p":
            model.pericardium.c_1 = value
        elif par == "c3_p":
            model.pericardium.c_3 = value
        elif par == "c4_p":
            model.pericardium.c_4 = value
        elif par == "prestretch":
            model.pericardium.pre_stretch = value

        # Heart area - maintain ratio of AmRefs within each wall but scale according to total AmRef given
        elif par == "amreflfw":
            model.heart.am_ref[model.heart.patches == 0] = model.heart.am_ref[model.heart.patches == 0] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 0])
        elif par == "amrefrfw":
            model.heart.am_ref[model.heart.patches == 1] = model.heart.am_ref[model.heart.patches == 1] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 1])
        elif par == "amrefsw":
            model.heart.am_ref[model.heart.patches == 2] = model.heart.am_ref[model.heart.patches == 2] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 2])
        elif par == "amrefla":
            model.heart.am_ref[model.heart.patches == 3] = model.heart.am_ref[model.heart.patches == 3] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 3])
        elif par == "amrefra":
            model.heart.am_ref[model.heart.patches == 4] = model.heart.am_ref[model.heart.patches == 4] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 4])

        # Wall volume, maintain current ratio in wall volumes between patches
        elif par == "vlfw":
            model.heart.vw[model.heart.patches == 0] = value * model.heart.vw[model.heart.patches == 0] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 0])
        elif par == "vrfw":
            model.heart.vw[model.heart.patches == 1] = value * model.heart.vw[model.heart.patches == 1] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 1])
        elif par == "vsw":
            model.heart.vw[model.heart.patches == 2] = value * model.heart.vw[model.heart.patches == 2] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 2])
        elif par == "vla":
            model.heart.vw[model.heart.patches == 3] = value * model.heart.vw[model.heart.patches == 3] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 3])
        elif par == "vra":
            model.heart.vw[model.heart.patches == 4] = value * model.heart.vw[model.heart.patches == 4] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 4])

        # Sigmoid parameters
        elif par == "fgmaxf+":
            model.growth.fgmax_f_plus = value
        elif par == "fgmaxf-":
            model.growth.fgmax_f_min = value
        elif par == "nf+":
            model.growth.n_f_plus = value
        elif par == "nf-":
            model.growth.n_f_min = value
        elif par == "s50f+":
            model.growth.s50_f_plus = value
        elif par == "s50f-":
            model.growth.s50_f_min = value
        elif par == "fgmaxr+":
            model.growth.fgmax_r_plus = value
        elif par == "fgmaxr-":
            model.growth.fgmax_r_min = value
        elif par == "nr+":
            model.growth.n_r_plus = value
        elif par == "nr-":
            model.growth.n_r_min = value
        elif par == "s50r+":
            model.growth.s50_r_plus = value
        elif par == "s50r-":
            model.growth.s50_r_min = value

        elif par == "t_mem":
            model.growth.t_mem = value

        # Growth parameters
        elif par == "tau_f-":
            model.growth.tau_f_min = value
        elif par == "tau_f_min":
            model.growth.tau_f_min = value
        elif par == "tau_f+":
            model.growth.tau_f_plus = value
        elif par == "tau_f_max":
            model.growth.tau_f_plus = value
        elif par == "tau_r-":
            model.growth.tau_r_min = value
        elif par == "tau_r+":
            model.growth.tau_r_plus = value
        elif par == "tau_r_min":
            model.growth.tau_r_min = value
        elif par == "tau_r_max":
            model.growth.tau_r_plus = value
        elif par == "theta_f_min":
            model.growth.theta_f_min = value
        elif par == "theta_f_max":
            model.growth.theta_f_max = value
        elif par == "theta_r_min":
            model.growth.theta_r_min = value
        elif par == "theta_r_max":
            model.growth.theta_r_max = value
        elif par == "gamma":
            model.growth.gamma = value

        ### The following metrics all use calculations to set wall volumes and areas, used for specific fitting schemes

        # Set total LV wall volume and distribute along left free wall and septal wall using patch number
        elif par == "lvwv":
            vw_tot_lv = np.sum(model.heart.vw[model.heart.patches == 0]) + np.sum(
                model.heart.vw[model.heart.patches == 2])
            for i_wall in [0, 2]:
                model.heart.vw[model.heart.patches == i_wall] = model.heart.vw[model.heart.patches == i_wall] * \
                                                                value / vw_tot_lv

        # Set midwall reference areas using ratio with left free wall
        elif par == "amrefrfwratio":
            am_ref_rfw = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
            model.heart.am_ref[model.heart.patches == 1] = model.heart.am_ref[model.heart.patches == 1] * \
                                                           am_ref_rfw / np.sum(model.heart.am_ref[model.heart.patches == 1])
        elif par == "amrefswratio":
            am_ref_sw = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
            model.heart.am_ref[model.heart.patches == 2] = model.heart.am_ref[model.heart.patches == 2] * \
                                                           am_ref_sw / np.sum(model.heart.am_ref[model.heart.patches == 2])
        elif par == "amreflaratio":
            model.heart.am_ref[model.heart.patches == 3] = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
        elif par == "amrefraratio":
            model.heart.am_ref[model.heart.patches == 4] = value * np.sum(model.heart.am_ref[model.heart.patches == 0])

        # Set wall volumes using ratio with left free wall
        elif par == "rfwvratio":
            vw_tot = value * np.sum(model.heart.vw[model.heart.patches == 0])
            model.heart.vw[model.heart.patches == 1] = model.heart.vw[model.heart.patches == 1] * \
                                                       vw_tot / np.sum(model.heart.vw[model.heart.patches == 1])
        elif par == "swvratio":
            vw_tot = value * np.sum(model.heart.vw[model.heart.patches == 0])
            model.heart.vw[model.heart.patches == 2] = model.heart.vw[model.heart.patches == 2] * \
                                                       vw_tot / np.sum(model.heart.vw[model.heart.patches == 2])
        elif par == "lawvratio":
            model.heart.vw[model.heart.patches == 3] = value * np.sum(
                model.heart.vw[model.heart.patches == 0])
        elif par == "rawvratio":
            model.heart.vw[model.heart.patches == 4] = value * np.sum(
                model.heart.vw[model.heart.patches == 0])

        # Set wall volumes based on specified wall thickness
        elif par == "lfwth":
            model.heart.vw[model.heart.patches == 0] = value * model.heart.am_ref[model.heart.patches == 0]
        elif par == "rfwth":
            model.heart.vw[model.heart.patches == 1] = value * model.heart.am_ref[model.heart.patches == 1]
        elif par == "swth":
            model.heart.vw[model.heart.patches == 2] = value * model.heart.am_ref[model.heart.patches == 2]
        elif par == "lawth":
            model.heart.vw[model.heart.patches == 3] = value * model.heart.am_ref[model.heart.patches == 3]
        elif par == "rawth":
            model.heart.vw[model.heart.patches == 4] = value * model.heart.am_ref[model.heart.patches == 4]

        # Update total wall volumes and areas to reflect any changes
        set_total_wall_volumes_areas(model)
        model.heart.v_tot_0 = unloaded_heart_volume(model.heart.am_ref_w, model.heart.vw_w)


def list_change_pars():
    """Return list of parameters that can be changed using change_pars()"""
    return ["sbv", "hr", "k_initial", "cvp", "cas", "cap", "cvs", "rvp", "rcs", "ras", "rvs", "rcp", "rap", "rav",
            "rmvb", "rtvb", "sact", "sfact", "tad", "td", "tr", "c1", "c3", "c4", "tact", "sact_a", "sfact_a", "tad_a",
            "td_a", "tr_a", "c1_a", "c3_a", "c4_a", "tact_a", "avd", "ivd_lv", "ivd_rv", "wth_p", "c1_p", "c3_p",
            "c4_p", "prestretch", "amreflfw", "amrefrfw", "amrefsw", "amrefla", "amrefra", "vlfw", "vrfw", "vsw",
            "vla", "vra", "fgmaxf+", "fgmaxf-", "nf+", "nf-", "s50f+", "s50f-", "fgmaxr+", "fgmaxr-", "nr+", "nr-",
            "s50r+", "s50r-", "t_mem", "tau_f-", "tau_f_min", "tau_f+", "tau_f_max", "tau_r-", "tau_r+", "tau_r_min",
            "tau_r_max", "theta_f_min", "theta_f_max", "theta_r_min", "theta_r_max", "gamma", "lvwv", "amrefrfwratio",
            "amrefswratio", "amreflaratio", "amrefraratio", "rfwvratio", "swvratio", "lawvratio", "rawvratio", "lfwth",
            "rfwth", "swth", "lawth", "rawth"]


def get_pars(model, pars):
    """Return dictionary of values for the parameters in the list pars"""

    # Change all list elements to lowercase
    par_orig = pars
    pars = [par.lower() for par in pars]

    values = {}

    # Heart parameters, ventricles and atria (with suffix _a) separately
    chamber_names = ["", "a"]
    i_ventricles = model.heart.patches == 0
    i_atria = model.heart.patches == 3
    i_chambers = [i_ventricles, i_atria]

    # Circulation
    for par_name in pars:

        # Circulation
        if par_name == "sbv":
            values[par_name] = model.circulation.sbv
        elif par_name == "hr":
            values[par_name] = model.circulation.hr
        elif par_name == "k_initial":
            values[par_name] = model.circulation.k

        # Capacitances
        elif par_name == "cvp":
            values[par_name] = model.capacitances.cvp
        elif par_name == "cas":
            values[par_name] = model.capacitances.cas
        elif par_name == "cap":
            values[par_name] = model.capacitances.cap
        elif par_name == "cvs":
            values[par_name] = model.capacitances.cvs

        # Resistances
        elif par_name == "rvp":
            values[par_name] = model.resistances.rvp
        elif par_name == "rcs":
            values[par_name] = model.resistances.rcs
        elif par_name == "ras":
            values[par_name] = model.resistances.ras
        elif par_name == "rvs":
            values[par_name] = model.resistances.rvs
        elif par_name == "rcp":
            values[par_name] = model.resistances.rcp
        elif par_name == "rap":
            values[par_name] = model.resistances.rap
        elif par_name == "rav":
            values[par_name] = model.resistances.rav
        elif par_name == "rmvb":
            values[par_name] = model.resistances.rmvb
        elif par_name == "rtvb":
            values[par_name] = model.resistances.rtvb

        elif par_name == "sact":
            values[par_name] = model.heart.sf_act[i_ventricles==0][0]
        elif par_name == "sfact":
            values[par_name] = model.heart.sf_act[i_ventricles==0][0]
        elif par_name == "tad":
            values[par_name] = model.heart.t_ad[i_ventricles==0][0]
        elif par_name == "td":
            values[par_name] = model.heart.tau_d[i_ventricles==0][0]
        elif par_name == "tr":
            values[par_name] = model.heart.tau_r[i_ventricles==0][0]
        elif par_name == "c1":
            values[par_name] = model.heart.c_1[i_ventricles==0][0]
        elif par_name == "c3":
            values[par_name] = model.heart.c_3[i_ventricles==0][0]
        elif par_name == "c4":
            values[par_name] = model.heart.c_4[i_ventricles==0][0]
        elif par_name == "tact":
            values[par_name] = model.heart.t_act[i_ventricles < 3]

        elif par_name == "sact_a":
            values[par_name] = model.heart.sf_act[i_ventricles==3][0]
        elif par_name == "sfact_a":
            values[par_name] = model.heart.sf_act[i_ventricles==3][0]
        elif par_name == "tad_a":
            values[par_name] = model.heart.t_ad[i_ventricles==3][0]
        elif par_name == "td_a":
            values[par_name] = model.heart.tau_d[i_ventricles==3][0]
        elif par_name == "tr_a":
            values[par_name] = model.heart.tau_r[i_ventricles==3][0]
        elif par_name == "c1_a":
            values[par_name] = model.heart.c_1[i_ventricles==3][0]
        elif par_name == "c3_a":
            values[par_name] = model.heart.c_3[i_ventricles==3][0]
        elif par_name == "c4_a":
            values[par_name] = model.heart.c_4[i_ventricles==3][0]
        elif par_name == "tact_a":
            values[par_name] = model.heart.t_act[i_ventricles >= 3]

        # Timing properties, atriaventricular delay and intraventricular delay (between lfw/rfw and septum)
        elif par_name == "avd":
            values[par_name] = model.heart.t_act[i_ventricles] - model.heart.t_act[i_atria]
        elif par_name == "ivd_lv":
            values[par_name] = model.heart.t_act[model.heart.patches == 0] - model.heart.t_act[model.heart.patches == 2]
        elif par_name == "ivd_rv":
            values[par_name] = model.heart.t_act[model.heart.patches == 1] - model.heart.t_act[model.heart.patches == 2]

        # Pericardium
        elif par_name == "wth_p":
            values[par_name] = model.pericardium.thickness
        elif par_name == "c1_p":
            values[par_name] = model.pericardium.c_1
        elif par_name == "c3_p":
            values[par_name] = model.pericardium.c_3
        elif par_name == "c4_p":
            values[par_name] = model.pericardium.c_4
        elif par_name == "prestretch":
            values[par_name] = model.pericardium.pre_stretch

        # Heart area - maintain ratio of AmRefs within each wall but scale according to total AmRef given
        elif par_name == "amreflfw":
            values[par_name] = np.sum(model.heart.am_ref[model.heart.patches == 0])
        elif par_name == "amrefrfw":
            values[par_name] = np.sum(model.heart.am_ref[model.heart.patches == 1])
        elif par_name == "amrefsw":
            values[par_name] = np.sum(model.heart.am_ref[model.heart.patches == 2])
        elif par_name == "amrefla":
            values[par_name] = np.sum(model.heart.am_ref[model.heart.patches == 3])
        elif par_name == "amrefra":
            values[par_name] = np.sum(model.heart.am_ref[model.heart.patches == 4])

        # Wall volume, maintain current ratio in wall volumes between patches
        elif par_name == "vlfw":
            values[par_name] = np.sum(model.heart.vw[model.heart.patches == 0])
        elif par_name == "vrfw":
            values[par_name] = np.sum(model.heart.vw[model.heart.patches == 1])
        elif par_name == "vsw":
            values[par_name] = np.sum(model.heart.vw[model.heart.patches == 2])
        elif par_name == "vla":
            values[par_name] = np.sum(model.heart.vw[model.heart.patches == 3])
        elif par_name == "vra":
            values[par_name] = np.sum(model.heart.vw[model.heart.patches == 4])

        # Sigmoid growth parameters
        elif par_name == "fgmaxf+":
            values[par_name] = model.growth.fgmax_f_plus
        elif par_name == "fgmaxf-":
            values[par_name] = model.growth.fgmax_f_min
        elif par_name == "nf+":
            values[par_name] = model.growth.n_f_plus
        elif par_name == "nf-":
            values[par_name] = model.growth.n_f_min
        elif par_name == "s50f+":
            values[par_name] = model.growth.s50_f_plus
        elif par_name == "s50f-":
            values[par_name] = model.growth.s50_f_min
        elif par_name == "fgmaxr+":
            values[par_name] = model.growth.fgmax_r_plus
        elif par_name == "fgmaxr-":
            values[par_name] = model.growth.fgmax_r_min
        elif par_name == "nr+":
            values[par_name] = model.growth.n_r_plus
        elif par_name == "nr-":
            values[par_name] = model.growth.n_r_min
        elif par_name == "s50r+":
            values[par_name] = model.growth.s50_r_plus
        elif par_name == "s50r-":
            values[par_name] = model.growth.s50_r_min

        elif par_name == "t_mem":
            values[par_name] = model.growth.t_mem

        # Kuhl growth parameters
        elif par_name == "tau_f-":
            values[par_name] = model.growth.tau_f_min
        elif par_name == "tau_f+":
            values[par_name] = model.growth.tau_f_plus
        elif par_name == "tau_f+":
            values[par_name] = model.growth.tau_f_plus
        elif par_name == "tau_r-":
            values[par_name] = model.growth.tau_r_min
        elif par_name == "tau_r+":
            values[par_name] = model.growth.tau_r_plus
        elif par_name == "theta_f_min":
            values[par_name] = model.growth.theta_f_min
        elif par_name == "theta_f_max":
            values[par_name] = model.growth.theta_f_max
        elif par_name == "theta_r_min":
            values[par_name] = model.growth.theta_r_min
        elif par_name == "theta_r_max":
            values[par_name] = model.growth.theta_r_max
        elif par_name == "gamma":
            values[par_name] = model.growth.gamma

        else:
            print(f"Parameter {par_name} not found in model")
            values[par_name] = np.nan

        # Restore original case of parameter names
        values = {par_orig[i]: value for i, (key, value) in enumerate(values.items())}

        # Convert all values to float
        values = {key: float(value) for key, value in values.items()}

    return values
