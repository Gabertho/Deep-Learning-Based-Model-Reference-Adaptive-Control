import numpy as np

def initialization():
    #Inicialização para controlador PD (Feedback Linearization)
    # Estados iniciais (posição e velocidade)
    x, y, z, fi = 0, 0, 0, 0
    dx, dy, dz, dfi = 0, 0, 0, 0
    # Criação de vetor de estados.
    q = np.array([x, y, z, fi])
    dq = np.array([dx, dy, dz, dfi])
    d2q = np.zeros(4)

    # Matriz de controle virtual
    v = np.zeros(4)

    # Definições dos ganhos do controlador PD
    Kp = np.diag([50, 50, 50, 50])
    Kd = np.diag([15, 15, 15, 15])

    return q, dq, d2q, v, Kp, Kd

