import numpy as np
import pandas as pd
import heart


def get_vitals(model, volumes, pressures, time, time_g=0):
    # Valve closing: valve is closed = 1
    valve_events = get_valve_events(pressures)
    wall_thickness, a_m_w = monarch_PimOomen.heart.get_wall_thickness(model)

    # ES
    es_frame = valve_events['av_closes']
    esp = pressures[es_frame, 2]  # [mmHg]
    esv = volumes[es_frame, 2]  # [mL]
    rv_esv = volumes[es_frame, 6]  # [mL]

    # ED Volume (when MV opens)
    ed_frame = valve_events['mv_closes']
    edp = pressures[ed_frame, 2]  # [mmHg]
    edv = volumes[ed_frame, 2]  # [mL]
    rv_edv = volumes[ed_frame, 6]  # [mL]

    # Maximum pressure
    p_max = max(pressures[:, 2])
    dpdt_max = max(np.gradient(pressures[:, 2], time[1] - time[0]))  # [mmHg/s]

    # Isovolumetric contraction duration
    ivc = abs(valve_events['mv_closes'] - valve_events['av_opens']) * model.solver.dt * 1000  # (ms)

    # ED wall thickness
    edwt = wall_thickness[ed_frame]
    edwa = a_m_w[ed_frame]

    # Cardiac function
    v_stroke = edv - esv  # [mL]
    ef = (edv - esv) / edv  # [-]
    co = v_stroke / time[-1] * 60 / 1e3  # [L/min]
    hr = 60 / time[-1]  # [s]

    # Arterial function
    map = pressures[:, 3].mean()
    esp_art = max(pressures[:, 3])
    edp_art = min(pressures[:, 3])

    # Turn into pandas
    vitals = pd.DataFrame([[edv, esv, edp, esp, ed_frame, es_frame, p_max, dpdt_max, v_stroke, ef, co,
                            map, hr, edp_art, esp_art, rv_edv, rv_esv, edwt[0], edwt[2], edwa[0], edwa[2],
                            ivc], ],
                          columns=['EDV', 'ESV', 'EDP', 'ESP', 'IED', 'IES', 'P_max', 'dp/dt_max', 'V_stroke', 'EF',
                                   'CO', 'MAP', 'HR', 'EDP_a', 'ESP_a', 'RV_EDV', 'RV_ESV', 'LFW_EDWT', 'SW_EDWT', 'LFW_A', 'SW_A', 'IVC'],
                          index=[time_g])

    return vitals


def get_valve_events(p):
    # Valve closing: valve is closed = 1, open = 0
    valve_events = {
        "mv_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 2] > p[:, 1], 1)), 0)),
        "mv_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 2] < p[:, 1], 1)), 0)),
        "av_closes": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 3] > p[:, 2], 1)), 0)),
        "av_opens": p.shape[0] - 1 - np.argmax(np.flip(np.diff(np.multiply(p[:, 3] < p[:, 2], 1)), 0))
    }
    return valve_events
