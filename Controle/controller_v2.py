import numpy as np

def feedback_linearization(q, d2qd, dqd, u):
    """
    Realiza a linearização por realimentação.

    Args:
    q (np.array): Estado atual do quadricóptero.
    d2qd (np.array): Aceleração desejada.
    dqd (np.array): Velocidade desejada.
    u (np.array): Sinal de controle do controlador PD.

    Returns:
    np.array: Vetor de controle linearizado.
    """
    # Substitua pelos valores corretos de mass e Izz
    mass = 0.389
    Izz = 0.002054

    M = np.array([[mass, 0, 0, 0],
                  [0, mass, 0, 0],
                  [0, 0, mass, 0],
                  [0, 0, 0, Izz]])

    v_fb = M @ (u + d2qd)
    return v_fb

def controller_v2(control, Kp, Kd, q, dqd, d2qd, q_til, dq_til, K, r, Kr, agent):
    """
    Controlador PD com Feedback Linearization.

    Args:
    control (int): Tipo de controle.
    Kp (np.array): Ganho proporcional.
    Kd (np.array): Ganho derivativo.
    q (np.array): Estado atual do quadricóptero.
    dqd (np.array): Velocidade desejada.
    d2qd (np.array): Aceleração desejada.
    q_til (np.array): Erro de posição.
    dq_til (np.array): Erro de velocidade.

    Returns:
    np.array: Vetor de controle.
    """
    if control == 1:  # Feedback Linearization
        u = -Kp @ q_til - Kd @ dq_til
        v = feedback_linearization(q, d2qd, dqd, u)

        vmax = 1  # Máximo input para o modelo do drone
        v = vmax * np.tanh(v / vmax)

        return v, u
    
    elif control == 2: #DMRAC
        u_pd = K @ q # K  = matriz de ganho (definir c Inoue)
        u_crm = Kr @ r #Kr = definir c Inoue valor.
        v_ad = agent.mrac_Cntrl(q, r)

        u = u_crm + u_pd - v_ad

        return u 