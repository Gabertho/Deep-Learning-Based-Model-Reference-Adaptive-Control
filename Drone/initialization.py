import numpy as np

from Controle.GPoff import GPoff

def initialization(uncertain):
    # Valores do controle Hinfinito
    h_t11 = 0.197990
    h_t12 = 0.949526
    h_ru = 0.02

    # Inicialização de posição e velocidade
    x = y = z = fi = 0
    dx = dy = dz = dfi = 0
    q = np.array([x, y, z, fi])
    dq = np.array([dx, dy, dz, dfi])
    d2q = np.zeros(4)

    # Matriz de controle virtual
    v = np.zeros(4)

    # Valores de THETA_GP
    THETA_GP = np.array([
        0.612879500245117,
        0.182171338091438,
        0.260273000706138,
        0.457826272500228
    ])

    # Valores finais desejados
    qf = np.array([5, 5, 3, np.pi / 4])
    dqf = np.zeros(4)
    d2qf = np.zeros(4)

    # Trajectory settings
    traj_opt = 1
    to = 0
    tf = 0
    ax = ay = az = afi = 0

    #Definições pré-definidas dos controladores 
    #------------------------------- Feedback Linearization -----------------%

    # Matrizes de controle Kp e Kd
    Kp = np.diag([50, 50, 50, 50])
    Kd = np.diag([15, 15, 15, 15])
    
    #--------------------------------- H infinito ----------------------------%

    # Matrizes T11, T12, etc.
    T11 = h_t11 * np.eye(4)
    T12 = h_t12 * np.eye(4)
    T11_inv = np.linalg.inv(T11)
    T_zero = np.block([[T11, T12], [np.zeros((4, 4)), np.eye(4)]])
    R = h_ru * np.eye(4)
    R_inv = np.linalg.inv(R)
    
    
    #------------------------- Rede Neural Adaptativa ------------------------%

    # Valores de THETA para rede neural adaptativa
    THETA = np.array([
        0.049424539731510,
        0.061707295451636,
        0.000461925433276,
        0.084625949075640,
        -0.016795187320992,
        0.033830358165730,
        0.194154634138103,
        0.182778602748309,
        0.012818117397264,
        0.092727957537253,
        0.149779259989359,
        0.033765763435889,
        0.140992308309683,
        0.137906576754882,
        0.124481892922731,
        0.011478649662710,
        0.075406168428600,
        0.158245482247798,
        0.110618521134542,
        0.055980525947054,
        0.101501855994308,
        0.032678259167206,
        0.091563683432064,
        0.165588211089541,
        0.146271708852501,
        0.055041114197478,
        0.131968142153905,
        0.074970661079984
    ])

    #--------------------------GP configurações Gerais------------------------%

    sigma_quad = 1e-5
    l = 10

    # Carregar dados de arquivos (assumindo que os arquivos existam)
    target = np.loadtxt('Drone/F0_v.txt')
    Xtrain = np.loadtxt('Drone/qe_v.txt')
    Xtest = np.loadtxt('Drone/qed_v.txt')

    # Chamada para função GPoff (a ser definida em Python)
    F0_gp, cov_gp = GPoff(sigma_quad, l, target.T, Xtrain, Xtest)

    
    #------------------------- GP com THETA Adaptativo - Artigo -----------------------
    Zt = Xtest[:, 2] if 'Xtest' in locals() else np.array([])
    yt = target[:, 2] if 'target' in locals() else np.array([])

    qe = np.zeros(4)
    qed = np.zeros(4)

    etol = 0.1
    p_max = 30

    omega_quad = 1  # ruído branco

    # Retornar todas as variáveis inicializadas
    return (h_t11, h_t12, h_ru, q, dq, d2q, v, THETA_GP, qf, dqf, d2qf, traj_opt, to, tf, ax, ay, az, afi, Kp, Kd,
            T11, T12, T11_inv, T_zero, R, R_inv, THETA, sigma_quad, l, F0_gp, cov_gp, Zt, yt, etol, p_max, omega_quad, qe, qed)
