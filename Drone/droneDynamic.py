import numpy as np

def droneDynamic(t, dt, q, dq, v, tempo_voo, uncertain, wind_pct):
    """
    Modela a dinâmica de um drone com perturbações e incertezas.

    Args:
    t (float): Tempo atual.
    dt (float): Intervalo de tempo.
    q (np.array): Estado atual do drone (posição e orientação).
    dq (np.array): Velocidade atual do drone.
    v (np.array): Sinal de controle do drone.
    tempo_voo (float): Tempo total de voo.
    uncertain (float): Fator de incerteza no modelo.
    wind_pct (float): Percentual do vento para cálculo de perturbação.

    Returns:
    tuple: Aceleração, velocidade e posição atualizadas do drone, vetor de perturbação e matriz B.
    """
    Izz = 0.002054
    mass = 0.389

    # Matriz de escala
    s_f1 = np.array([[4.33, 0, 0, 0],
                     [0, 4.12, 0, 0],
                     [0, 0, 4.42, 0],
                     [0, 0, 0, 5.92]])

    mass_fixed = mass + mass * 1.5
    mass = mass + mass * uncertain
    Izz_fixed = Izz + Izz * 1.5
    Izz = Izz + Izz * uncertain

    # Cálculo de perturbações induzidas pelo vento
    psi = q[3]
    vmax_wind = 1  # m/s vento fraco

    v_wind = vmax_wind * np.sin(0.2 * t)
   

    input_max = 5
    dmax = (wind_pct / 100) * input_max
    dx = dmax * np.sin(t - 4)**2 if 4 < t < 7 else 0
    dy = dmax * np.sin(t - 9)**2 if 9 < t < 12 else 0
    dz = dmax * np.sin(t - 14)**2 if 14 < t < 17 else 0
    dpsi = 0

    d = np.array([dx, dy, dz, dpsi])

    q[3] = normalize_angle(q[3], -np.pi)

    M_fixed = np.array([[mass_fixed, 0, 0, 0],
                        [0, mass_fixed, 0, 0],
                        [0, 0, mass_fixed, 0],
                        [0, 0, 0, Izz_fixed]])
    M = np.array([[mass, 0, 0, 0],
                  [0, mass, 0, 0],
                  [0, 0, mass, 0],
                  [0, 0, 0, Izz]])
    B = np.linalg.inv(M)
    B_fixed = np.linalg.inv(M_fixed)

    d2q = B @ (s_f1 @ v) + (B_fixed @ d)
    dq = d2q * dt + dq
    q = dq * dt + q

    return d2q, dq, q, d, B

def normalize_angle(angle, low):
    """
    Normaliza um ângulo para um intervalo semi-fechado.

    Args:
    angle (float): Ângulo em radianos.
    low (float): Valor inicial do intervalo de normalização.

    Returns:
    float: Ângulo normalizado.
    """
    return angle - 2 * np.pi * np.floor((angle - low) / (2 * np.pi))
