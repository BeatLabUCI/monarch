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
    lab_f = lsc[iv] / lsref[iv] # stretch of contractile element
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

    # Initial estimate of tensions
    tx, ty, tmw, rm, xm = tension_numba(vs, ys, vml, vmr, amw0, dadtw)
    err = err0 = np.linalg.norm(np.array([tx, ty]))

    # Newton parameters
    err_max, iter_max = 1e-3, 10  # Increased max iterations
    vs0, ys0 = vs, ys
    damping = 1.0  # Initial damping factor
    min_damping = 0.1
    trust_radius = 0.1  # Initial trust region radius
    eps = 1e-8  # Small number to prevent division by zero

    # For adaptive step size in Jacobian calculation
    h_vs = max(1e-6, 1e-4 * abs(vs))
    h_ys = max(1e-6, 1e-4 * abs(ys))

    for iter in range(iter_max):
        if err <= err_max:
            break

        # Calculate Jacobian with improved numerical precision
        txv, tyv, _, _, _ = tension_numba(vs + h_vs, ys, vml, vmr, amw0, dadtw)
        txy, tyy, _, _, _ = tension_numba(vs, ys + h_ys, vml, vmr, amw0, dadtw)

        dtxdv, dtydv = (txv - tx) / h_vs, (tyv - ty) / h_vs
        dtxdy, dtydy = (txy - tx) / h_ys, (tyy - ty) / h_ys

        # Add regularization to Jacobian if needed
        reg = 1e-6 * (abs(dtxdv) + abs(dtydy)) / 2.0
        dtxdv += reg
        dtydy += reg

        detj = dtxdv * dtydy - dtxdy * dtydv

        # Compute Newton step with safeguards
        if abs(detj) > eps:
            dv = (-dtydy * tx + dtxdy * ty) / detj
            dy = (+dtydv * tx - dtxdv * ty) / detj

            # Apply trust region constraint
            step_norm = np.sqrt(dv ** 2 + dy ** 2)
            if step_norm > trust_radius * np.sqrt(vs ** 2 + ys ** 2):
                scale = trust_radius * np.sqrt(vs ** 2 + ys ** 2) / step_norm
                dv *= scale
                dy *= scale
        else:
            # Fallback to gradient descent if Jacobian is singular
            dv = -damping * tx * 1e-2
            dy = -damping * ty * 1e-2

        # Apply damping
        dv *= damping
        dy *= damping

        # Update solution and evaluate new error
        vs_new, ys_new = vs + dv, ys + dy
        tx_new, ty_new, tmw_new, rm_new, xm_new = tension_numba(vs_new, ys_new, vml, vmr, amw0, dadtw)
        err_new = np.linalg.norm(np.array([tx_new, ty_new]))

        # Implement improved line search with Armijo condition
        if err_new < err:
            # Step is good - accept it and possibly increase trust region
            vs, ys = vs_new, ys_new
            tx, ty, tmw, rm, xm = tx_new, ty_new, tmw_new, rm_new, xm_new
            err = err_new

            # Update trust region and damping based on improvement
            reduction_ratio = (err0 - err) / (err0 * 0.5)  # Expected vs actual reduction
            if reduction_ratio > 0.75:
                trust_radius = min(2.0 * trust_radius, 0.5)
                damping = min(1.0, damping * 1.2)
            elif reduction_ratio < 0.25:
                trust_radius = max(trust_radius * 0.5, 0.01)
                damping = max(min_damping, damping * 0.8)

            err0, vs0, ys0 = err, vs, ys
        else:
            # Step is bad - decrease trust region and try again
            trust_radius *= 0.5
            damping = max(min_damping, damping * 0.5)

            # If trust region becomes too small, use bisection
            if trust_radius < 1e-4:
                alpha = 0.5  # Try halfway between previous and rejected point
                vs_new = vs0 + alpha * dv
                ys_new = ys0 + alpha * dy
                tx_new, ty_new, tmw_new, rm_new, xm_new = tension_numba(vs_new, ys_new, vml, vmr, amw0, dadtw)
                err_new = np.linalg.norm(np.array([tx_new, ty_new]))

                if err_new < err0:
                    vs, ys = vs_new, ys_new
                    tx, ty, tmw, rm, xm = tx_new, ty_new, tmw_new, rm_new, xm_new
                    err = err_new

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
    sigp = np.minimum(2 * c_1 * (lab_f_slice2 - 1) + c_3 * (np.exp(c_4 * (lab_f_slice2 - 1)) - 1), 1e10)
    dsigpdlab_f = np.minimum(4 * c_1 * lab_f_slice + 2 * c_3 * c_4 * lab_f_slice * np.exp(c_4 * (lab_f_slice2 - 1)),
                             1e10)

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


def get_wall_thickness(model):
    """
    Calculate the thickness and midwall reference area of each wall throughout the cardiac cycle
    """
    # Midwall area
    lab_f_w = np.transpose(np.array([np.mean(model.heart.lab_f[:, model.heart.patches == i],
                                             axis=1) for i in range(0, 5)]))
    a_m_w = model.heart.am_ref_w * (lab_f_w ** 2)
    # Wall thickness
    wall_thickness = model.heart.vw_w / a_m_w
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
    model.heart.ys = r_s

    # vs is the volume of the spherical cap
    model.heart.vs = 1 / 6 * np.pi * h_s * (3 * r_s ** 2 + h_s ** 2)
