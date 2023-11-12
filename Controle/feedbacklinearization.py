import numpy as np

def normalize_angle(angle):
    """
    Normaliza um ângulo para o intervalo [-pi, pi].

    Args:
    angle (float): Ângulo a ser normalizado.

    Returns:
    float: Ângulo normalizado.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def feedback_linearization(q, d2q_d, u):
    """
    Realiza a linearização por realimentação.

    Args:
    q (np.array): Estado atual do quadricóptero.
    d2q_d (np.array): Aceleração desejada.
    u (np.array): Sinal de controle do controlador PD.

    Returns:
    np.array: Vetor de controle linearizado.
    """
    Izz = 0.002054
    mass = 0.389

    fi = normalize_angle(q[3])

    M = np.array([[mass, 0, 0, 0],
                  [0, mass, 0, 0],
                  [0, 0, mass, 0],
                  [0, 0, 0, Izz]])

    v_fb = M @ (u + d2q_d)

    return v_fb

# Agora, você pode chamar esta função dentro da função controller_pd
