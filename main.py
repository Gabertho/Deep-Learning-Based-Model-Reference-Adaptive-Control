import numpy as np
from Drone.droneDynamic import droneDynamic
from Drone.initialization import initialization
from Drone.normalize_angle_f import normalize_angle_f
from Plots.plot3dim_control import plot3dim_control
from Trajetoria.trajectory_desired import trajectory_desired
from Controle.controller_v2 import controller_v2

def main():
    # Configurações iniciais
    control = 1  # Escolha o controle
    traj_des = 1  # Escolha a trajetória

    # Parâmetros da trajetória
    vel_d = 0.3  # Velocidade desejada em m/s
    A = 1  # Amplitude da trajetória
    omg = vel_d / A  # Velocidade angular desejada em rad/s
    tempo_voo = 20  # Tempo de voo em segundos
    dt = 0.01
    N = int(tempo_voo / dt)

    # Incerteza
    uncertain = 1.5
    wind_pct = 70

    # Inicialização
    initialization_vars = initialization(uncertain)
    (h_t11, h_t12, h_ru, q, dq, d2q, v, THETA_GP, qf, dqf, d2qf, traj_opt, to, tf, ax, ay, az, afi, Kp, Kd, T11, T12, T11_inv, T_zero, R, R_inv, THETA, sigma_quad, l, F0_gp, cov_gp, Zt, yt, etol, p_max, omega_quad, qe, qed) = initialization_vars


    
    # Preparação dos arrays para armazenamento de dados
    t_v = np.zeros(N)
    u_v = np.zeros((4, N))
    v_v = np.zeros((4, N))
    us_v = np.zeros((4, N))
    F0_v = np.zeros((4, N))
    deltaF0_v = np.zeros((4, N))
    d_v = np.zeros((4, N))
    B_v = np.zeros((4, 4, N))
    dB_v = np.zeros((4, N))
    q_v = np.zeros((4, N))
    dq_v = np.zeros((4, N))
    d2q_v = np.zeros((4, N))
    qe_v = np.zeros((4, N))
    qd_v = np.zeros((4, N))
    dqd_v = np.zeros((4, N))
    d2qd_v = np.zeros((4, N))
    qed_v = np.zeros((4, N))
    q_til_v = np.zeros((4, N))
    dq_til_v = np.zeros((4, N))
    d2q_til_v = np.zeros((4, N))

    for i in range(N):
        t = i * dt
        t_v[i] = t

        # Planejamento de trajetória
        
        (qd, dqd, d2qd, d3qd, to, tf, ax, ay, az, afi) = trajectory_desired(traj_des, q, dq, d2q, qf, dqf, d2qf, t, dt, traj_opt, vel_d, A, omg, to, tf, ax, ay, az, afi)


        # Cálculo do erro da posição, velocidade e aceleração
        q_til = q[:3] - qd[:3]
        q_til = np.append(q_til, normalize_angle_f(q[3] - qd[3], -np.pi))
        dq_til = dq - dqd
        d2q_til = d2q - d2qd
        
        if (i == 0):
            print("Valores após trajectory_desired na 1a iteracao:")
            print("qd:", qd)
            print("dqd:", dqd)
        
        if (i == 1):
            print("Valores após trajectory_desired na 2a iteracao:")
            print("qd:", qd)
            print("dqd:", dqd)

        # Controlador
        (v, u, F0, deltaF0, THETA_GP, THETA, us, Zt, yt, v_raw) = controller_v2(control, Kp, Kd, q, dqd, d2qd, R_inv, T_zero, T11, T11_inv, q_til, dq_til, dq, qd, THETA, qe, qed, h_t11, h_t12, h_ru, F0_gp, sigma_quad, l, THETA_GP, Zt, yt, etol, p_max, omega_quad, i, dt, uncertain)
      
        # Dinâmica do Drone
        (d2q, dq, q, d, B) = droneDynamic(t, dt, q, dq, v, tempo_voo, uncertain, wind_pct)
     
        # Armazenando dados para plotagem
        q_v[:, i] = q
        dq_v[:, i] = dq
        d2q_v[:, i] = d2q
        qd_v[:, i] = qd
        dqd_v[:, i] = dqd
        d2qd_v[:, i] = d2qd
        u_v[:, i] = u
        v_v[:, i] = v
        us_v[:, i] = us
        F0_v[:, i] = F0
        deltaF0_v[:, i] = deltaF0
        d_v[:, i] = d
        B_v[:, :, i] = B
        dB_v[:, i] = B @ d
        qe_v[:, i] = qe
        qed_v[:, i] = qed
        q_til_v[:, i] = q_til
        dq_til_v[:, i] = dq_til
        d2q_til_v[:, i] = d2q_til

    # Plotagem da trajetória
    plot3dim_control(qd_v, q_v, control)
    
if __name__ == "__main__":
    main()