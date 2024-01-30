import numpy as np

from Controle.GP_adaptativo_art import gp_adaptativo_art
from Controle.GP_theta_adaptativo import gp_theta_adaptativo
from Controle.Rede_Neural import rede_neural
from Controle.droneNominalModel import droneNominalModel
from Controle.feedbacklinearization import feedback_linearization

def controller_v2(control, Kp, Kd, q, dqd, d2qd, R_inv, T_zero, T11, T11_inv, q_til, dq_til, dq, qd, THETA, qe, qed, h_t11, h_t12, h_ru, F0_gp, sigma_quad, l, THETA_GP, Zt, yt, etol, p_max, omega_quad, i, dt, uncertain):
    F0 = droneNominalModel(d2qd, dq_til, dqd, q_til, q, h_t11, h_t12, h_ru)

    if control == 1:  # Feedback Linearization
        u = -Kp @ q_til - Kd @ dq_til
        v = feedback_linearization(q, d2qd, dqd, u)
        v_raw = v
        vmax = 1  # max parrot input
        v[:4] = vmax * np.tanh(v[:4] / vmax)
        us = np.zeros(4)
        deltaF0 = np.zeros(4)
        Zt = 0
        yt = 0
        THETA = 0
        THETA_GP = np.zeros(4)

    elif control == 2:  # H infinito
        u = -R_inv @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til])
        v = F0 + u
        vmax = 1  # max parrot input
        v[:4] = vmax * np.tanh(v[:4] / vmax)
        E = 0
        us = np.zeros(4)
        deltaF0 = np.zeros(4)
        Zt = 0
        yt = 0
        THETA = 0
        THETA_GP = np.zeros(4)

    elif control == 3:  # H infinito com Rede Neural
        u = -R_inv @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til])
        E, THETA, us = rede_neural(THETA, dt, i, q_til, dq_til, T_zero, T11, q, dq, qd, dqd, d2qd)
        v = F0 + E * THETA + T11_inv @ u
        vmax = 1  # max parrot input
        v[:4] = vmax * np.tanh(v[:4] / vmax)
        deltaF0 = E * THETA
        Zt = 0
        yt = 0
        THETA_GP = np.zeros(4)

    elif control == 4:  # H infinito com GP THETA Adaptativo
        u = -R_inv @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til])
        mu_GPtheta, THETA_GP, us = gp_theta_adaptativo(sigma_quad, l, THETA_GP, dt, i, q_til, dq_til, T_zero, T11, qe.T, qed)
        v = F0 + mu_GPtheta + T11_inv @ u + us
        vmax = 1  # max parrot input
        v[:4] = vmax * np.tanh(v[:4] / vmax)
        deltaF0 = mu_GPtheta
        Zt = 0
        yt = 0
        THETA = 0

    elif control == 5:  # H infinito com GP Offline
        u = -R_inv @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til])
        v = F0 + F0_gp[i, :].T + T11_inv @ u
        vmax = 1  # max parrot input
        v[:4] = vmax * np.tanh(v[:4] / vmax)
        deltaF0 = F0_gp[i, :]
        us = np.zeros(4)
        Zt = 0
        yt = 0
        THETA = 0
        THETA_GP = np.zeros(4)

    elif control == 6:  # H infinito com GP Adaptativo Artigo
        u = -R_inv @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til])
        u[:4] = np.tanh(u[:4])
        zt_1 = qed
        mu, cov, Zt, zt_1, yt = gp_adaptativo_art(sigma_quad, l, zt_1, Zt, yt, i, omega_quad, etol, p_max)
        v = F0 + mu[:, i] + T11_inv @ u
        yt[:, i + 1] = mu[:, i]
        Zt[:, len(Zt[0, :]) + 1] = qe
        stdv = np.sqrt(np.diag(cov))
        deltaF0 = mu[:, i]
        us = np.zeros(4)
        THETA = 0
        THETA_GP = np.zeros(4)

    else:
        raise ValueError("Controle inexistente")

    return v, u, F0, deltaF0, THETA_GP, THETA, us, Zt, yt, v_raw
