import numpy as np

from Controle.kernel import kernel

def gp_theta_adaptativo(sigma_quad, l, THETA_GP, T, i, q_til, dq_til, T_zero, T11, qe, qed):
    K_sx = np.zeros(4)

    for j in range(4):
        K_sx[j] = kernel(sigma_quad, qed[j], qe[j], l, 1)

    Z = 0.5 * np.eye(4)
    N1 = 1000
    delta = 10

    K_sx = np.diag(K_sx)

    B2 = np.vstack([np.eye(4), np.zeros((4, 4))])
    aux_THETAd = -np.linalg.inv(Z) @ K_sx @ T11 @ B2.T @ T_zero @ np.vstack([dq_til, q_til])

    if (np.dot(THETA_GP.T, THETA_GP) <= N1) or (np.dot(THETA_GP.T, THETA_GP) > N1 and np.dot(THETA_GP.T, aux_THETAd) <= 0):
        THETA_d = aux_THETAd
    else:
        THETA_d = aux_THETAd - ((((np.dot(THETA_GP.T, THETA_GP) - N1) * np.dot(THETA_GP.T, aux_THETAd)) / (delta * np.dot(THETA_GP.T, THETA_GP))) * THETA_GP)

    if i > 2:
        THETA_GP = THETA_GP + THETA_d * T

    mult_q_til = np.dot(q_til.T, q_til)
    kxe = (1 / 0.2) * np.sqrt(abs(mult_q_til))

    sgn = np.sign(T11 @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til]))

    us = -kxe * sgn

    mu = K_sx @ THETA_GP

    return mu, THETA_GP, us
