import numpy as np

from Drone.normalize_angle_f import normalize_angle_f

def feedback_linearization(q, d2q_d, dq_d, u):
    Izz = 0.002054
    mass = 0.389

    fi = normalize_angle_f(q[3], -np.pi)  # Normalização do ângulo

    M = np.array([
        [mass, 0, 0, 0],
        [0, mass, 0, 0],
        [0, 0, mass, 0],
        [0, 0, 0, Izz]
    ])

    N = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    v_fb = M @ (u + d2q_d)

    return v_fb
