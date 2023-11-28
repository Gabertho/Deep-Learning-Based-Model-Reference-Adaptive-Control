import numpy as np
from droneDynamic import droneDynamic

from initialization import initialization
from plot3dim_control import plot3dim_control
from trajectory_desired import trajectory_desired
from controller_v2 import controller_v2

import tensorflow as tf

def main():
    # Configurações iniciais
    control = 1  # Controlador PD com Feedback Linearization
    traj_des = 1   # Trajetória Elipse (1) ou Trajetória Ponto a ponto (2)
    vel_d = 0.3   # Velocidade desejada (m/s)
    A = 1         # Amplitude da trajetória
    omg = vel_d / A  # Velocidade angular desejada em rad/s. (v=wr - > w = v/r))
    tempo_voo = 20 #em segundos.

    # Incerteza
    uncertain = 1.5 
    wind_pct = 0     # Percentual do vento 

    # Tempo
    dt = 0.01
    N = int(tempo_voo / dt)

    # Inicialização
    q, dq, d2q, v, Kp, Kd = initialization()

    # Valores finais desejados e parâmetros de trajetória
    qf = np.array([5, 5, 5, 5])  # Estado final desejado
    dqf = np.array([0, 0, 0, 0])  # Velocidade final desejada
    d2qf = np.array([0, 0, 0, 0])  # Aceleração final desejada
    ax, ay, az, afi = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    to, tf = 0, 10000  # Tempos inicial e final

    # Arrays para armazenar dados para plotagem
    q_v = np.zeros((4, N))
    qd_v = np.zeros((4, N))

    # Loop principal
    for i in range(N):
        t = i * dt

        # Planejamento de trajetória
        qd, dqd, d2qd, d3qd, to, tf = trajectory_desired(
            traj_des, q, dq, d2q, qf, dqf, d2qf, t, dt, vel_d, A, omg, to, tf, ax, ay, az, afi, i
        )

        # Erro de posição e velocidade
        q_til = q - qd
        dq_til = dq - dqd

        # Aplicando o controlador PD ou DMRAC
        v, u = controller_v2(control, Kp, Kd, q, dqd, d2qd, q_til, dq_til)

        # Simulação da dinâmica do drone
        d2q, dq, q, d, B = droneDynamic(t, dt, q, dq, v, tempo_voo, uncertain, wind_pct)

        # Armazenando dados para plotagem
        q_v[:, i] = q
        qd_v[:, i] = qd

    # Plotagem da trajetória
    plot3dim_control(qd_v, q_v, control)

if __name__ == "__main__":
    main()
