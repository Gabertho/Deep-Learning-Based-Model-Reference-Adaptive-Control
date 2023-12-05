import numpy as np
from Drone.droneDynamic import droneDynamic

from Drone.initialization import initialization
from Plots.plot3dim_control import plot3dim_control
from Trajetoria.trajectory_desired import trajectory_desired
from Controle.controller_v2 import controller_v2
from Controle.dmrac import MRAC
from refLibrary import refSignal
from refmodel import refModel
import tensorflow as tflow

def main():
    # Configurações iniciais
    control = 2  # Controlador PD com Feedback Linearization
    traj_des = 2  # Trajetória Elipse (1) ou Trajetória Ponto a ponto (2)
    vel_d = 0.3   # Velocidade desejada (m/s)
    A = 1         # Amplitude da trajetória
    omg = vel_d / A  # Velocidade angular desejada em rad/s. (v=wr - > w = v/r))
    tempo_voo = 20 #em segundos.
    K = 0; #Matriz de ganho p DMRAC.
    r = 0; #sinal de referência
    Kr = 0; #Matriz de ganho p DMRAC.
    # Incerteza
    uncertain = 1.5 
    wind_pct = 0    # Percentual do vento 

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

    #Definições DMRAC
    start_state = np.reshape([1,1],(2,1)) 
    ref_env = refModel(start_state)
    ref_cmd = refSignal(N)


    #Rde neural 
    with tflow.Session() as sess:
        agent = MRAC(sess,4,4)
        sess.run(tflow.global_variables_initializer())
        ref_cmd.stepCMD()
        n_idx = 0

        pos_rec = [start_state[0]]
        ref_pos_rec = [start_state[0]]
        vel_rec = [start_state[1]]
        ref_vel_rec = [start_state[1]]
        ref_rec = [0]
        # Loop principa
        for i in range(N):
            t = i * dt

            # Planejamento de trajetória
            qd, dqd, d2qd, d3qd, to, tf = trajectory_desired(
                traj_des, q, dq, d2q, qf, dqf, d2qf, t, dt, vel_d, A, omg, to, tf, ax, ay, az, afi, i
            )

            # Erro de posição e velocidade
            q_til = q - qd
            dq_til = dq - dqd

            # Aplicando o controlador PD
            if control == 1:
                v, u = controller_v2(control, Kp, Kd, q, dqd, d2qd, q_til, dq_til, 0, 0, 0, 0)
            #Aplicando o controlador DMRAC
            elif control == 2:
                adap_cntrl = agent.total_Cntrl(q, ref_env.state, ref_cmd.refsignal[n_idx])
                ref_env.stepRefModel(ref_cmd.refsignal[n_idx])
                n_idx = n_idx+1
                v, u = controller_v2(control, Kp, Kd, q, dqd,d2q, q_til, dq_til, K, ref_cmd.refsignal[n_idx], Kr, agent)

            # Simulação da dinâmica do drone
            d2q, dq, q, d, B = droneDynamic(t, dt, q, dq, v, tempo_voo, uncertain, wind_pct)

            # Armazenando dados para plotagem
            q_v[:, i] = q
            qd_v[:, i] = qd

    # Plotagem da trajetória
    plot3dim_control(qd_v, q_v, control)

if __name__ == "__main__":
    main()