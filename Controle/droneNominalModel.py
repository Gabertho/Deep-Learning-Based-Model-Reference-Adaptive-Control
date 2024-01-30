from Drone.normalize_angle_f import normalize_angle_f
import numpy as np


def droneNominalModel(d2q_d, dq_til, dq_d, q_til, q, h_t11, h_t12, h_ru):
    Izz = 0.002054
    mass = 0.389

    T11 = h_t11 * np.eye(4)
    T12 = h_t12 * np.eye(4)

    q[3] = normalize_angle_f(q[3], -np.pi)  # Ajuste do índice para base-0 do Python

    fi = q[3]  # Normalização do ângulo

    M = np.array([
        [mass, 0, 0, 0],
        [0, mass, 0, 0],
        [0, 0, mass, 0],
        [0, 0, 0, Izz]
    ])

    T11_inv = np.linalg.inv(T11)
    M_inv = np.linalg.inv(M)

    F0 = M @ (d2q_d - T11_inv @ T12 @ dq_til)

    return F0