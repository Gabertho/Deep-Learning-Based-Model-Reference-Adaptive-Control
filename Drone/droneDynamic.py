import numpy as np

from Drone.normalize_angle_f import normalize_angle_f


def droneDynamic(t, dt, q, dq, v, tempo_voo, uncertain, wind_pct):
    m = 0.5
    Izz = 0.002054
    mass = 0.389

    # Matriz de escala
    s_f1 = np.array([[4.33, 0, 0, 0],
                     [0, 4.12, 0, 0],
                     [0, 0, 4.42, 0],
                     [0, 0, 0, 5.92]])

    massfixed = mass + mass * 1.5
    mass = mass + mass * uncertain
    Izzfixed = Izz + Izz * 1.5
    Izz = Izz + Izz * uncertain

    psi = q[3]

    # Velocidade induzida pelo vento
    vmax_wind = 1  # m/s vento fraco
    v_wind = vmax_wind * np.sin(0.2 * t)

    # Perturbação induzida pelo vento
    dx = dy = dz = 0
    dmax = (wind_pct / 100) * 5

    if 4 < t < 7:
        dx = dmax * np.sin(t - 4) ** 2
    if 9 < t < 12:
        dy = dmax * np.sin(t - 9) ** 2
    if 14 < t < 17:
        dz = dmax * np.sin(t - 14) ** 2

    dpsi = 0
    d = np.array([dx, dy, dz, dpsi])

    q[3] = normalize_angle_f(q[3], -np.pi)

    Mfixed = np.array([[massfixed, 0, 0, 0],
                       [0, massfixed, 0, 0],
                       [0, 0, massfixed, 0],
                       [0, 0, 0, Izzfixed]])
    M = np.array([[mass, 0, 0, 0],
                  [0, mass, 0, 0],
                  [0, 0, mass, 0],
                  [0, 0, 0, Izz]])
    B = np.linalg.inv(M)
    Bfixed = np.linalg.inv(Mfixed)

    # Cálculo da dinâmica do drone
    d2q = B @ (s_f1 @ v) + (Bfixed @ d)
    dq = d2q * dt + dq
    q = dq * dt + q

    return d2q, dq, q, d, B