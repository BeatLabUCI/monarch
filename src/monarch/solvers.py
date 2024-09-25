import numpy as np
from numba import njit

from .heart import guess_vs_ys, set_total_wall_volumes_areas, v2p_numba, unloaded_heart_volume
from .import_export import load_converged_sol


def initialize_solvers_volumes(model, use_converged):
    """
    Initialize solver parameters and compartmental volumes. It is called either at the start of a single heart beat
    simulation or at the start of a growth simulation.
    """

    # Initialize timing
    set_time_vector(model)

    # Use converged solution if present
    if use_converged and model.converged_file.is_file():
        load_converged_sol(model)

        # If amount of patches is different, do not use initial values for Lsc and C
        if model.heart.patches.size != model.heart.lsc.size:
            model.heart.lsc = model.heart.ls_ref - model.heart.ls_eiso           # [um]
            model.heart.c = np.zeros(model.heart.patches.shape)                  # [-]

    # Set initial volume estimate
    model.volumes[0, :] = model.circulation.k * model.circulation.sbv              # [mL]

    # Get initial guesses for Vs and Ys based on LV volume at t=0 and ventricular geometry
    if not use_converged or not model.converged_file.is_file():
        guess_vs_ys(model)
        model.heart.dv = 0.01*model.heart.vs
        model.heart.dy = 0.01*model.heart.ys

    # Calculate total heart wall volume and midwall reference area
    set_total_wall_volumes_areas(model)

    # Compute reference volume of heart for pericardial mechanics computations [mm^3]
    model.heart.v_tot_0 = unloaded_heart_volume(model.heart.am_ref_w, model.heart.vw_w)

    # Use predefined initial guesses if no converged solution is found or if using a different number if patches
    # than in previous run. In the latter case the converged volume guesses (if present) will still be used
    if (not use_converged) or (not model.converged_file.is_file()):
        # If no previously converged solution is found, set state variables based on parameter values
        model.heart.lsc = model.heart.ls_ref - model.heart.ls_eiso           # [um]
        model.heart.c = np.zeros(model.heart.patches.shape)                  # [-]


def set_time_vector(model):
    """
    Sets time vector
    """
    # Set time vector - rerun this function if changing HR after model initialisation
    model.time = np.linspace(0, 60/model.circulation.hr, model.solver.n_inc)
    model.solver.dt = model.time[1] - model.time[0]


def rk4_wrapper(model):
    """
    Because numba currently at point of writing doesn't support classes, I am writing all parameters needed for these
    functions into non-class variables.
    Everything within the rk4_loop is compiled by numba and much faster than a pure python implementation. Everything
    outside the rk4_loop is rewriting the class parameters into non-class variables that can be processed by the numba
    compiled code.
    """
    # Calls
    n_inc = model.solver.n_inc
    dt = model.solver.dt
    m_time = model.time

    # The numba package can read the resistances fast from a numpy array, but can't read them from a class object.
    # Therefore, I will write the 9 resistances into a numpy array.
    r = np.array([model.resistances.rap,   # r[0]
                  model.resistances.ras,   # r[1]
                  model.resistances.rav,   # r[2]
                  model.resistances.rcp,   # r[3]
                  model.resistances.rcs,   # r[4]
                  model.resistances.rmvb,  # r[5]
                  model.resistances.rtvb,  # r[6]
                  model.resistances.rvp,   # r[7]
                  model.resistances.rvs])  # r[8]

    # All pressures
    p = model.pressures

    # All volumes
    v = model.volumes

    # The numba package can read the resistances fast from a numpy array, but can't read them from a class object.
    # Therefore, I will write the 9 capacitances into a numpy array.
    cap = np.array([model.capacitances.cvp,  # cap[0]
                    model.capacitances.cas,  # cap[1]
                    model.capacitances.cvs,  # cap[2]
                    model.capacitances.cap])  # cap[3]

    # Calls
    vs = model.heart.vs
    ys = model.heart.ys
    dv = model.heart.dv
    dy = model.heart.dy

    # MultiPatch stiffness and unloaded area, for ventricles only
    iv = model.heart.i_ventricles
    vw_iv = model.heart.vw[iv]
    amref_iv = model.heart.am_ref[iv]
    lsref = model.heart.ls_ref

    # Kinematics for Ls = Lsc
    lsc = model.heart.lsc
    lsc_0 = model.heart.lsc_0
    ls_eiso = model.heart.ls_eiso
    lab_f = model.heart.lab_f

    # Stress and stress derivative,
    heart_c = model.heart.c
    c_1 = model.heart.c_1
    c_3 = model.heart.c_3
    c_4 = model.heart.c_4

    # Time constants
    t_act = model.heart.t_act
    t_ad = model.heart.t_ad
    tr = model.heart.tau_r * t_ad
    # Other implementation of td used in Oomen 2021 that assumes repolarization is not length-dependent was not copied
    # into this code. See older version of this implementation of td.
    td = model.heart.tau_d * t_ad  # Assume repolarization is length-dependent - used in Walmsley 2015

    # Active stress
    sf_act = model.heart.sf_act

    # Contractile element length
    v_max = model.heart.v_max

    # All total stresses and its derivatives
    sig_f = model.heart.sig_f

    # Wall stiffness and unloaded area
    patches_iv = model.heart.patches[iv]

    # Enclosed midwall volumes of L and R free walls
    vw_w = model.heart.vw_w

    # All heart.am, heart.xm and heart.ys_store
    rm = model.heart.rm
    xm = model.heart.xm
    y_store = model.heart.ys_store

    # Atrial kinematics and tension
    ia = model.heart.i_atria
    vw_ia = model.heart.vw[ia]
    amref_ia = model.heart.am_ref[ia]

    # pericardium
    p_thick = model.pericardium.thickness
    p_lab_f = model.pericardium.lab_f
    v_tot_0 = model.heart.v_tot_0
    p_lab_pre = model.pericardium.lab_pre
    p_c_1 = model.pericardium.c_1
    p_c_3 = model.pericardium.c_3
    p_c_4 = model.pericardium.c_4
    p_pressure = model.pericardium.pressure

    # This is the routine, within which everything is compiled into machine code by the numba package.
    v, p, inc, vs, ys, dv, dy, lsc, lab_f, heart_c, sig_f, dsigdlab_f, rm, xm, y_store, p_lab_f, p_pressure = (
        rk4_loop(n_inc, dt, m_time, r, p, v, cap, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref, lsc, lsc_0, ls_eiso, lab_f,
                 heart_c, t_act, t_ad, tr, td, sf_act, c_1, c_3, c_4, v_max, sig_f, patches_iv, vw_w, rm, xm,
                 y_store, vw_ia, amref_ia, ia, p_thick, p_lab_f, v_tot_0, p_lab_pre, p_c_1, p_c_3, p_c_4, p_pressure))

    # Here I write the outputs of the rk4_loop back into the class structure, so that all other functions work as usual.
    model.volumes = v
    model.pressures = p
    model.solver.inc = inc
    model.heart.vs = vs
    model.heart.ys = ys
    model.heart.dv = dv
    model.heart.dy = dy
    model.heart.lsc = lsc
    model.heart.lab_f = lab_f
    model.heart.c = heart_c
    model.heart.sig_f = sig_f
    model.heart.dsigdlab_f = dsigdlab_f
    model.heart.rm = rm
    model.heart.xm = xm
    model.heart.ys_store = y_store
    model.pericardium.lab_f = p_lab_f
    model.pericardium.pressure = p_pressure


@njit(cache=False)
def rk4_loop(n_inc, dt, m_time, r, p, v, cap, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref, lsc, lsc_0, ls_eiso, lab_f,
             heart_c, t_act, t_ad, tr, td, sf_act, c_1, c_3, c_4, v_max, sig_f, patches_iv, vw_w, rm, xm, y_store, vw_ia,
             amref_ia, ia, p_thick, p_lab_f, v_tot_0, p_lab_pre, p_c_1, p_c_3, p_c_4, p_pressure):
    # Here I only pass the volumes of the previous iteration into the rk4_step function to reduce overhead.
    v_slice_before = v[0, :]
    # Here I only pass the pressure of the previous iteration into the rk4_step function to reduce overhead.
    p_slice_before = p[0, :]
    for inc in range(n_inc):
        # Here I only pass the lab_f of the current iteration into the rk4_step function to reduce overhead.
        lab_f_slice = lab_f[inc, :]

        # Time constant
        tc = (m_time[inc] * 1e3 - t_act)
        # Since the model runs one cardiac cycle in a loop, the computation of the contractility in
        # heart.get_stress_myocardium_numba should wrap around until the next patch is activated. The modulo operator is
        # doing exactly that.

        # Here I only pass the totel stress of the current iteration into the rk4_step function to reduce overhead.
        sig_f_slice = sig_f[inc, :]

        # Here I only pass the heart.rm, heart.xm and heart.y_store of the current iteration into the rk4_step function
        # to reduce overhead.
        rm_slice = rm[inc, :]
        xm_slice = xm[inc, :]

        # Run the current rk4 step
        (v_slice_now, p_slice_now, vs, ys, dv, dy, lsc, lab_f_slice, heart_c, sig_f_slice, dsigdlab_f, rm_slice,
         xm_slice, y_store_slice, p_lab_f_slice, p_pressure_slice) = (
            rk4_step(inc, dt, tc, r, p_slice_before, v_slice_before, cap, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref,
                     lsc, lsc_0, ls_eiso, lab_f_slice, heart_c, t_ad, tr, td, sf_act, c_1, c_3, c_4, v_max, sig_f_slice,
                     patches_iv, vw_w, rm_slice, xm_slice, vw_ia, amref_ia, ia, p_thick, v_tot_0, p_lab_pre, p_c_1,
                     p_c_3, p_c_4))

        # Update volumes
        v[inc, :] = v_slice_now
        v_slice_before = v_slice_now

        # Update pressures
        p[inc, :] = p_slice_now
        p_slice_before = p_slice_now

        # Update lab_f
        lab_f[inc, :] = lab_f_slice

        # Update sig_f
        sig_f[inc, :] = sig_f_slice

        # Update rm, xm and ys_store
        rm[inc, :] = rm_slice
        xm[inc, :] = xm_slice
        y_store[inc] = y_store_slice

        # Update pericardium.lab_f
        p_lab_f[inc] = p_lab_f_slice

        # Update pericardium.pressure
        p_pressure[inc] = p_pressure_slice

    return v, p, inc, vs, ys, dv, dy, lsc, lab_f, heart_c, sig_f, dsigdlab_f, rm, xm, y_store, p_lab_f, p_pressure


@njit(cache=False)
def rk4_step(inc, dt, tc, r, p_slice_before, v_slice_now, cap, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref, lsc, lsc_0,
             ls_eiso, lab_f_slice, heart_c, t_ad, tr, td, sf_act, c_1, c_3, c_4, v_max, sig_f_slice, patches_iv, vw_w,
             rm_slice, xm_slice, vw_ia, amref_ia, ia, p_thick, v_tot_0, p_lab_pre, p_c_1, p_c_3, p_c_4):
    """
    Code engine: 4th order Runge-Kutta differential equation solver to calculate
    volumes and pressures at the current time point.
    The argument v_slice_now that is passed is actually the volume of the previous slice that will be updated for all
    but the first increment. We call it v_slice_now to save an else statement after the "if inc:" statement.
    """

    # Compartments:
    # Vp  LA  LV  AS  Vs  RA  RV  Ap
    # 0   1   2   3   4   5   6   7

    # k1-4
    k1 = dv_combined(p_slice_before, r)
    k2 = dv_combined(p_slice_before + 0.5 * dt * k1, r)
    k3 = dv_combined(p_slice_before + 0.5 * dt * k2, r)
    k4 = dv_combined(p_slice_before + dt * k3, r)

    # Compute new volumes
    # The volume is not computed for the first iteration with inc = 0, because another equation is used in the
    # just_beat_it function. For inc = 1, 2, ... the following computation will be performed.
    if inc:
        v_slice_now = v_slice_now + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

    # Compute new pressures of blood vessels and heart chambers
    p_slice_now = vessels_v2p_numba(v_slice_now, p_slice_before, cap)

    # Atrial kinematics and tension
    vma = np.maximum(np.array([v_slice_now[1], v_slice_now[5]]) * 1e3 + 0.5 * vw_ia, 0.5 * vw_ia)
    rma = ((3 * vma) / (4 * np.pi)) ** (1 / 3)
    ama = 3 * vma / rma

    (vs, ys, dv, dy, lsc, lab_f_slice, heart_c, sig_f_slice, dsigdlab_f, rm_slice, xm_slice, y_store_slice,
     p_lab_f_slice, p_pressure_slice, p_slice_now) = (
        v2p_numba(dt, tc, vs, ys, dv, dy, iv, vw_iv, amref_iv, lsref, lsc, lsc_0, ls_eiso, lab_f_slice, heart_c,
                        t_ad, tr, td, sf_act, c_1, c_3, c_4, v_max, sig_f_slice, patches_iv, rm_slice, xm_slice,
                        amref_ia, ia, ama, rma, vw_ia, v_slice_now, vw_w, p_thick, v_tot_0, p_lab_pre, p_c_1, p_c_3,
                        p_c_4, p_slice_now))

    return (v_slice_now, p_slice_now, vs, ys, dv, dy, lsc, lab_f_slice, heart_c, sig_f_slice, dsigdlab_f, rm_slice,
            xm_slice, y_store_slice, p_lab_f_slice, p_pressure_slice)


@njit(cache=False)
def vessels_v2p_numba(v_now, p_before, cap):
    """
    Calculate pressure in blood vessel compartments
    Capacitances:
    cap[0]: capacitance cvp
    cap[1]: capacitance cas
    cap[2]: capacitance cvs
    cap[3]: capacitance cap
    """
    return np.array([v_now[0]/cap[0],   # p_now[0]
                     p_before[1],     # p_now[1]
                     p_before[2],     # p_now[2]
                     v_now[3]/cap[1],   # p_now[3]
                     v_now[4]/cap[2],   # p_now[4]
                     p_before[5],     # p_now[5]
                     p_before[6],     # p_now[6]
                     v_now[7]/cap[3]])  # p_now[7]


@njit(cache=False)
def dv_combined(p, r):
    """
    Compile all volume changes
    Pressures:
    p[0]: pressure in pulmonary veins
    p[1]: pressure in left atrium
    p[2]: pressure in left ventricle
    p[3]: pressure in systemic arterial
    p[4]: pressure in systemic veins
    p[5]: pressure in right atrium
    p[6]: pressure in right ventricle
    p[7]: pressure in pulmonary arteries
    Resistances:
    r[0]: resistance rap - pulmonary arterial resistance
    r[1]: resistance ras - systemic arterial resistance
    r[2]: resistance rav - atrial ventricular resistance
    r[3]: resistance rcp - pulmonary characteristic resistance
    r[4]: resistance rcs - systemic characteristic resistance
    r[5]: resistance rmvb - mitral valve backflow resistance
    r[6]: resistance rtvb - tricuspid valve backflow resistance
    r[7]: resistance rvp - pulmonary venous resistance
    r[8]: resistance rvs - systemic venous resistance
    """
    return np.array([(p[7] - p[0]) / r[0]
                     - (p[0] - p[1]) / r[7],  # pulmonary veins
                     (p[0] - p[1]) / r[7] \
                     - (p[1] - p[2]) / r[2] * (p[1] > p[2])
                     + (p[2] - p[1]) / r[5] * (p[2] > p[1]),  # left atria
                     (p[1] - p[2]) / r[2] * (p[1] > p[2])
                     - (p[2] - p[3]) / r[4] * (p[2] > p[3])
                     - (p[2] - p[1]) / r[5] * (p[2] > p[1]),  # left ventricle
                     (p[2] - p[3]) / r[4] * (p[2] > p[3])
                     - (p[3] - p[4]) / r[1],  # systemic arterial
                     (p[3] - p[4]) / r[1]
                     - (p[4] - p[5]) / r[8],  # systemic veins
                     (p[4] - p[5]) / r[8]
                     - (p[5] - p[6]) / r[2] * (p[5] > p[6])
                     + (p[6] - p[5]) / r[6] * (p[6] > p[5]),  # right atrium
                     (p[5] - p[6]) / r[2] * (p[5] > p[6])
                     - (p[6] - p[7]) / r[3] * (p[6] > p[7])
                     - (p[6] - p[5]) / r[6] * (p[6] > p[5]),  # right ventricle
                     (p[6] - p[7]) / r[3] * (p[6] > p[7])
                     - (p[7] - p[0]) / r[0]])  # pulmonary arteries
