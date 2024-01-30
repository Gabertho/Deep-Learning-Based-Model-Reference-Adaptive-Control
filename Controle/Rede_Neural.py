import numpy as np

def rede_neural(THETA, T, i, q_til, dq_til, T_zero, T11, q, dq, qd, dqd, d2qd):
    pk = 7  # número de neurônios
    n = 4   # número de redes neurais
    en = 20 # número de entradas

    # Pesos
    W = np.array([-1] * 8 + [1] * 12)  # Padrão para todas as redes e neurônios
    W = np.tile(W, (n, pk, 1))

    # Bias
    B = np.array([[-3, -2, -1, 0, 1, 2, 3]] * n)

    qe = np.concatenate([q, dq, qd, dqd, d2qd])

    E = np.zeros((n, pk * n))
    for k in range(n):
        for j in range(pk):
            sumwq = np.sum(W[k, j, :] * qe)
            H = np.tanh(sumwq + B[k, j])
            E[k, j + pk * k] = H

    # Parâmetros do algoritmo
    Z = 0.2 * np.eye(pk * n)
    N1 = 1000
    delta = 10

    B2 = np.vstack([np.eye(4), np.zeros((4, 4))])
    aux_THETAd = -np.linalg.inv(Z) @ E.T @ T11 @ B2.T @ T_zero @ np.vstack([dq_til, q_til])

    if (np.dot(THETA.T, THETA) <= N1) or (np.dot(THETA.T, THETA) > N1 and np.dot(THETA.T, aux_THETAd) <= 0):
        THETA_d = aux_THETAd
    else:
        THETA_d = aux_THETAd - (((np.dot(THETA.T, THETA) - N1) * np.dot(THETA.T, aux_THETAd)) / (delta * np.dot(THETA.T, THETA))) * THETA

    # Atualização dos parâmetros
    if i > 2:
        THETA = THETA + THETA_d * T

    mult_q_til = np.dot(q_til.T, q_til)
    kxe = (1 / 0.2) * np.sqrt(abs(mult_q_til))
    sgn = np.sign(T11 @ np.hstack([np.eye(4), np.zeros((4, 4))]) @ T_zero @ np.vstack([dq_til, q_til]))
    us = -kxe * sgn

    return E, THETA, us
