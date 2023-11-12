import numpy as np

def par_pol_f(to, tf, qo, dqo, d2qo, qf, dqf, d2qf):
    """
    Calcula os coeficientes de um polinômio de quinto grau.

    Args:
    to (float): Tempo inicial.
    tf (float): Tempo final.
    qo (float): Posição inicial.
    dqo (float): Velocidade inicial.
    d2qo (float): Aceleração inicial.
    qf (float): Posição final.
    dqf (float): Velocidade final.
    d2qf (float): Aceleração final.

    Returns:
    np.array: Coeficientes do polinômio.
    """
    q_v = np.array([qo, dqo, d2qo, qf, dqf, d2qf])
    delta_t = tf - to

    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [1, delta_t, delta_t**2, delta_t**3, delta_t**4, delta_t**5],
                  [0, 1, 2*delta_t, 3*delta_t**2, 4*delta_t**3, 5*delta_t**4],
                  [0, 0, 2, 6*delta_t, 12*delta_t**2, 20*delta_t**3]])
    
    return np.linalg.inv(A) @ q_v

