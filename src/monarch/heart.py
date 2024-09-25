import numpy as np
from numba import njit


@njit(cache=False)
def v2p_numba(dt, tc, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref, lsc, lsc_0, ls_eiso, lab_f_slice, heart_c, t_ad, tr,
              td, sf_act, c_1, c_3, c_4, v_max, sig_f_slice, patches_iv, rm_slice, xm_slice, amref_ia, ia, ama, rma,
              vw_ia, v_slice_now, vw_w, p_thick, v_tot_0, p_lab_pre, p_c_1, p_c_3, p_c_4, p_slice_now):
    """
    Solve TriSeg balance of forces and calculate ventricular pressures, adopted from Walmsley et al., PLOS Comp
    Biol 2015, and Lumens et al., Annals of Biomed Eng 2009. Total LV and RV volume is known at input.
    Calculates and updates C, Lsc, P, and sig
    """
    # Kinematics for Ls = Lsc
    lab_f = lsc[iv] / lsref[iv]
    lab_f_slice[iv] = lab_f

    # Stress and stress derivative, prevent update of C and Lsc
    _, _, sig_f_slice, dsigdlab_f = (
        get_stress_myocardium_numba(dt, tc, lab_f_slice, lsref, lsc, lsc_0, ls_eiso, t_ad, tr, td, heart_c, sf_act, c_1,
                                    c_3, c_4, v_max))

    am = lab_f ** 2 * amref_iv  # Eq 9
    tm = sig_f_slice[iv] * vw_iv / (2 * am)  # Eq 11

    # Patch stiffness and estimated unloaded area
    dadt = 4 * am ** 2 / vw_iv / (dsigdlab_f[iv] * lab_f - 2 * sig_f_slice[iv])  # Eq 13
    am0 = am - tm * dadt  # Eq 14

    # Wall stiffness and unloaded area, sum of all patches in each wall
    amw0 = np.array([sum(am0[patches_iv == i]) for i in range(0, 3)])
    dadtw = np.array([sum(dadt[patches_iv == i]) for i in range(0, 3)])

    # Enclosed midwall volumes of L and R free walls
    vml = v_slice_now[2] * 1e3 + 0.5 * (vw_w[0] + vw_w[2])
    vmr = v_slice_now[6] * 1e3 + 0.5 * (vw_w[1] + vw_w[2])

    # Initial estimate of tensions, if error is already below criterion the local solver will be skipped
    tx, ty, tmw, rm, xm = tension_numba(vs, ys, vml, vmr, amw0, dadtw)
    err = err0 = np.linalg.norm(np.array([tx, ty]))  # Get Newton-Raphson error

    # Initialize Newton scheme
    err_max, iter_max, iter = 1e-3, 3, 0
    vs0, ys0 = vs, ys

    while (err > err_max) and (iter < iter_max):
        # Perturb solution
        txv, tyv, _, _, _ = tension_numba(vs + dv, ys, vml, vmr, amw0, dadtw)
        txy, tyy, _, _, _ = tension_numba(vs, ys + dy, vml, vmr, amw0, dadtw)

        # Inverse Jacobian matrix and determinant
        dtxdv, dtydv, dtxdy, dtydy = (txv - tx) / dv, (tyv - ty) / dv, (txy - tx) / dy, (tyy - ty) / dy
        detj = dtxdv * dtydy - dtxdy * dtydv

        # Estimated change in vs and ys
        # If a derivative goes to 0, the solution become unstable and overshoots target values. Prevent this by
        # nudging the next guess away from the unstable point when dv or dy becomes unreasonably large.
        if not detj == 0:
            dv = (-dtydy * tx + dtxdy * ty) / detj
            dy = (+dtydv * tx - dtxdv * ty) / detj
            if np.absolute(dv) > 0.1 * np.absolute(vs):
                dv = np.sign(dv) * vs * 1e-2
            if np.absolute(dy) > 0.1 * np.absolute(ys):
                dy = np.sign(dy) * ys * 1e-2
        else:
            dv = np.sign(dv) * vs * 1e-2
            dy = np.sign(dy) * ys * 1e-2

        # Update solution and state variables
        vs, ys = vs + dv, ys + dy
        tx, ty, tmw, rm, xm = tension_numba(vs, ys, vml, vmr, amw0, dadtw)
        err = np.linalg.norm(np.array([tx, ty]))  # Get Newton-Raphson error

        # Update error if smaller, else perform line search to prevent overshooting minimum
        if err < err0:
            err0, vs0, ys0 = err, vs, ys
        else:
            iter_line = 0
            while (err > err0) and (iter_line < iter_max):
                tx0, ty0, _, _, _ = tension_numba(vs0, ys0, vml, vmr, amw0, dadtw)

                # Estimate best line position
                g0 = np.linalg.norm(np.array([tx0, ty0]))  # Get Newton-Raphson error
                g0prime = -g0 / np.linalg.norm(np.array([vs - vs0, ys - ys0]))
                g1 = np.linalg.norm(np.array([tx, ty]))  # Get Newton-Raphson error
                lab = max(-g0prime / (2 * (g1 - g0 - g0prime)), 0.1)

                # New estimates and error
                vs, ys = (1 - lab) * vs0 + lab * vs, (1 - lab) * ys0 + lab * ys
                tx, ty, tmw, rm, xm = tension_numba(vs, ys, vml, vmr, amw0, dadtw)
                err = np.linalg.norm(np.array([tx, ty]))  # Get Newton-Raphson error
                iter_line += 1

        iter += 1

    # Ventricular kinematics at force balance and assign new geometry to class
    am = am0 + tmw[patches_iv] * dadt
    lab_f_slice[iv] = np.sqrt(am / amref_iv)
    rm_slice[0:3], xm_slice[0:3], y_store_slice = rm, xm, ys

    # Assign and calculate atrial stretch
    lab_f_slice[ia] = np.sqrt(ama / amref_ia)
    rm_slice[3:5] = rma

    # Compute stress for all ventricular patches and atrial walls
    lsc, heart_c, sig_f_slice, dsigdlab_f = (
        get_stress_myocardium_numba(dt, tc, lab_f_slice, lsref, lsc, lsc_0, ls_eiso, t_ad, tr, td, heart_c, sf_act, c_1,
                                    c_3, c_4, v_max))

    # Compute atrial tension
    tma = vw_ia * sig_f_slice[ia] / (2 * ama)

    # Compute pericardial kinematics: total heart volume [mm^3] and pericardial stretch [0]
    v_heart = sum(np.array([v_slice_now[1], v_slice_now[2], v_slice_now[5], v_slice_now[6]])) * 1e3 + sum(vw_w)
    if p_thick > 0.0:
        p_lab_f_slice = (v_heart / v_tot_0) ** ((1 / 3) * p_lab_pre)
    else:
        p_lab_f_slice = 1.0  # Prevent potential crashes when not using pericardium

    # Pericardial pressure [mmHg], via thin-walled sphere theory
    p_lab_f2 = p_lab_f_slice ** 2
    sig_p = np.minimum(2 * p_c_1 * (p_lab_f2 - 1) + p_c_3 * (np.exp(p_c_4 * (p_lab_f2 - 1)) - 1), 1e20)  # Stress [MPa]
    r_p = (3 * v_heart) / (4 * np.pi) ** (1 / 3)
    p_pressure_slice = 2 * sig_p * p_thick / r_p * 7.5e3

    # Compute ventricular and atrial pressures
    p_slice_now = np.array([p_slice_now[0],
                            2 * tma[0] / rma[0] * 7.5e3 + p_pressure_slice,
                            2 * tmw[0] / np.absolute(rm[0]) * 7.5e3 + p_pressure_slice,
                            p_slice_now[3],
                            p_slice_now[4],
                            2 * tma[1] / rma[1] * 7.5e3 + p_pressure_slice,
                            2 * tmw[1] / np.absolute(rm[1]) * 7.5e3 + p_pressure_slice,
                            p_slice_now[7]])

    return (vs, ys, dv, dy, lsc, lab_f_slice, heart_c, sig_f_slice, dsigdlab_f, rm_slice, xm_slice, y_store_slice,
            p_lab_f_slice, p_pressure_slice, p_slice_now)


@njit(cache=False)
def get_stress_myocardium_numba(dt, tc, lab_f_slice, lsref, lsc, lsc_0, ls_eiso, t_ad, tr, td, heart_c, sf_act, c_1, c_3,
                                c_4, v_max):
    """
    Calculate myocardial stress, updates: sig(inc), C, Lsc, dsigdlab
    """
    # Calls (convert time from s to ms)
    dt = dt * 1e3

    # Stretch and sarcomere element lengths
    ls = lab_f_slice * lsref
    lsc_norm = lsc / lsc_0 - 1  # normalized contractile element length
    lse_norm = np.maximum((ls - lsc) / ls_eiso, 0.0001)  # Normalized series elastic element

    # Time constants
    ta = (0.65 + 1.0570 * lsc_norm) * t_ad
    t = tc / tr

    # Active stress
    sf_iso = heart_c * lsc_norm * lsc_0 * sf_act  # Isometric component
    siga, dsigadlab_f = sf_iso * lse_norm, (sf_iso * ls / lab_f_slice) / ls_eiso

    # Passive stress [MPa]
    lab_f_slice2 = lab_f_slice ** 2
    sigp = np.minimum(2 * c_1 * (lab_f_slice2 - 1) + c_3 * (np.exp(c_4 * (lab_f_slice2 - 1)) - 1), 1e20)
    dsigpdlab_f = np.minimum(4 * c_1 * lab_f_slice + 2 * c_3 * c_4 * lab_f_slice * np.exp(c_4 * (lab_f_slice2 - 1)),
                             1e20)

    # Length and time-dependent quantities to update C
    x = np.minimum(8, np.maximum(0, t))  # normalized time during rise of activation
    frise = x ** 3 * np.exp(-x) * (8 - x) ** 2 * 0.020  # rise of contraction, 'amount of Ca++ release'
    x = (tc - ta) / td  # normalized time during decay of activation
    gx = 0.5 + 0.5 * np.sin(np.sign(x) * np.minimum(np.pi / 2, np.absolute(x)))  # always>0
    fl = np.tanh(0.75 * 9.1204 * lsc_norm ** 2)  # regulates increase of contractility with Ls

    # State variable 1: contractile element length (prevent Lsc from going beyond reference length)
    lsc = np.maximum(lsc + dt * (lse_norm - 1) * v_max, 1.0001 * lsc_0)

    # State variable 2: electrical activation (only for non-ischemic patches)
    heart_c = heart_c + dt * (fl * frise / tr - heart_c * gx / td)

    # Total stress and its derivative
    sig_f_slice, dsigdlab_f = siga + sigp, dsigpdlab_f + dsigadlab_f

    return lsc, heart_c, sig_f_slice, dsigdlab_f


@njit(cache=False)
def tension_numba(vs, ys, vml, vmr, am0, dadt):
    """
    Wall volumes and area
    """

    # Adjust L and R spherical midwall volumes to satisfy VLV and VRV
    vm = np.array([-vml + vs, vmr + vs, vs])

    # Estimate xm from Eq. 9 from CircAdapt derivation
    # Solving 3rd order polynomial
    v = (3 / np.pi) * np.absolute(vm)
    q = (v + np.sqrt(v ** 2 + ys ** 6)) ** (1 / 3)
    xm = np.sign(vm) * (q - ys ** 2 / q)
    xm[xm == 0] = 0.001

    # calculate midwall area Am and curvature Cm=1/rm
    xm2, ym2 = xm ** 2, ys ** 2
    r2 = xm2 + ym2
    rm = r2 / (2 * xm)
    am = np.maximum(np.pi * r2, 1.001 * am0)

    # calculation of tension T and components Tx, Ty
    tm = (am - am0) / dadt
    txi, tyi = (ym2 - xm2) / r2 * tm, ys / rm * tm
    tref = np.sqrt(sum(tm ** 2))

    tx, ty = sum(txi) / tref, sum(tyi) / tref  # axial and radial tension component

    return tx, ty, tm, rm, xm


def get_wall_thickness(cg):
    """
    Calculate the thickness and midwall reference area of each wall throughout the cardiac cycle
    """
    # Midwall area
    lab_f_w = np.transpose(np.array([np.mean(cg.heart.lab_f[:, cg.heart.patches == i], axis=1) for i in range(0, 5)]))
    a_m_w = cg.heart.am_ref_w * (lab_f_w ** 2)
    # Wall thickness
    wall_thickness = cg.heart.vw_w / a_m_w
    return wall_thickness, a_m_w


def set_total_wall_volumes_areas(model):
    """
    Compute total wall volume and reference areas
    """
    model.heart.vw_w = np.zeros(5)
    model.heart.am_ref_w = np.zeros(5)
    model.heart.n_patches = np.zeros(5)
    for i in range(0, 5):
        model.heart.am_ref_w[i] = sum(model.heart.am_ref[model.heart.patches == i])
        model.heart.vw_w[i] = sum(model.heart.vw[model.heart.patches == i])
        model.heart.n_patches[i] = sum(model.heart.patches == i)


def heart_volume(model):
    """
    Get total heart volume (cavity plus wall volumes) at the current cardiac cycle time increment
    """
    return sum(model.volumes[model.solver.inc, [1, 2, 5, 6]]) * 1e3 + sum(model.heart.vw_w)


def unloaded_heart_volume(am_ref_w, vw_w):
    """
    Calculate total unloaded heart volume (cavity plus wall volumes) [mm^3]
    """
    return sum(unloaded_volumes(am_ref_w, vw_w)) * 1e3 + sum(vw_w)


def unloaded_volumes(am_ref_w, vw_w):
    """
    Calculate unloaded cavity volumes given reference midwall area and wall volume. Can only be used to estimate
    unloaded (and not loaded) volumes due to geometry assumptions made in the midwall volume calculations
    """

    # Preallocate midwall volume array
    v_m_0 = np.zeros(4)

    # Left ventricle: sphere with total surface Sw plus Lfw
    v_m_0[0] = (am_ref_w[0] + am_ref_w[2]) ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Right ventricle
    r = np.sqrt((am_ref_w[1] + am_ref_w[2]) / (4 * np.pi))  # Radius of RV if SW would have been directed towards the LV
    h = am_ref_w[2] / (2 * np.pi * r)  # Height of the septal wall spherical cap
    v_s = np.pi * h ** 2 / 3 * (3 * r - h)  # Septal cap midwall volume
    v_m_0[1] = (am_ref_w[1] + am_ref_w[2]) ** (3 / 2) / (
                6 * np.sqrt(np.pi)) - v_s  # Volume of RV minus septal cap volume

    # Left atrium, use spherical geometry
    v_m_0[2] = am_ref_w[3] ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Right atrium, use spherical geometry
    v_m_0[3] = am_ref_w[4] ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Calculate cavity volumes by subtracting half the wall volumes from midwall volumes and convert from mm^3 to mL
    v_0 = (v_m_0 - 0.5 * np.array([vw_w[0] + vw_w[2], vw_w[1] + vw_w[2], vw_w[3], vw_w[4]])) * 1e-3

    return v_0


def guess_vs_ys(model):
    """
    Estimate initial values for vs and ys based on initial LV cavity volume and wall thickness. This estimate assumes
    spherical geometry at the initial time point and a stretch of 1.05
    """

    # Get midwall area of the LV at the first time point
    v_m_lv = model.volumes[0, 2] + 0.5 * (model.heart.vw_w[0] + model.heart.vw_w[2])
    am_m_lv = np.pi ** (1 / 3) * (6 * v_m_lv) ** (2 / 3)

    # Estimate stretch at the first time point
    lab = np.sqrt(am_m_lv / (model.heart.am_ref_w[0] + model.heart.am_ref_w[2]))

    # Find spherical cap height that matches the midwall reference area of the septal wall initial stretch
    h_s = (model.heart.am_ref_w[2] * lab ** 2) / (2 * np.pi * ((3 * v_m_lv) / (4 * np.pi)) ** (1 / 3))

    # ys is twice the base radius of the spherical cap
    r_m = ((3 * v_m_lv) / (4 * np.pi)) ** (1 / 3)
    r_s = np.sqrt(h_s * (2 * r_m - h_s))
    model.heart.ys = 2 * r_s

    # vs is the volume of the spherical cap
    model.heart.vs = 1 / 6 * np.pi * h_s * (3 * r_s ** 2 + h_s ** 2)
