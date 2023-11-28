from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from refLibrary import refSignal
from refmodel import refModel
from wingrock import wingRock
from NeuralNetwork.controller import MRAC

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

def controller_v2(control, Kp, Kd, q, dqd, d2qd, q_til, dq_til):
    """
    Controlador PD com Feedback Linearization e DMRAC.

    Args:
    control (int): Tipo de controle.
    Kp (np.array): Ganho proporcional.
    Kd (np.array): Ganho derivativo.
    q (np.array): Estado atual do quadricóptero.
    dqd (np.array): Velocidade desejada.
    d2qd (np.array): Aceleração desejada.
    q_til (np.array): Erro de posição.
    dq_til (np.array): Erro de velocidade.
    Kr (np.array): Ganho para o termo feed-forward.
    r (np.array): Referência.
    K (np.array): Ganho para o termo feed-back.
    x (np.array): Estado atual.
    mrac (MRAC): Objeto MRAC.

    Returns:
    np.array: Vetor de controle.
    """
    if control == 1:  # Feedback Linearization
        u = -Kp @ q_til - Kd @ dq_til
        v = feedback_linearization(q, d2qd, dqd, u)

        vmax = 1  # Máximo input para o modelo do drone
        v = vmax * np.tanh(v / vmax)

        return v, u
    elif control == 2:  # DMRAC
        sim_endTime = 100
        start_state = np.reshape([1,1],(2,1))
        env = wingRock(start_state)
        ref_env = refModel(start_state)
        N = int(sim_endTime/env.timeStep)
        ref_cmd = refSignal(N)

        with tf.Session() as sess:
            agent = MRAC(sess,2,1,10)

            sess.run(tf.global_variables_initializer())

            ref_cmd.stepCMD()
            n_idx = 0

            pos_rec = [start_state[0]]
            ref_pos_rec = [start_state[0]]
            vel_rec = [start_state[1]]
            ref_vel_rec = [start_state[1]]
            ref_rec = [0]

            for idx in range(0, N):
                adap_cntrl = agent.total_Cntrl(env.state, ref_env.state, ref_cmd.refsignal[n_idx])
                env.applyCntrl(adap_cntrl)
                ref_env.stepRefModel(ref_cmd.refsignal[n_idx])
                pos_rec.append(env.state[0])
                ref_pos_rec.append(ref_env.state[0])
                vel_rec.append(env.state[1])
                ref_vel_rec.append(ref_env.state[1])
                ref_rec.append(ref_cmd.refsignal[n_idx])
                n_idx = n_idx+1


            u = agent.total_Cntrl

            v = feedback_linearization(q, d2qd, dqd, u)

            plt.figure(1)
            ax1 = plt.subplot(211)
            plt.plot(pos_rec, color='red', label='$x(t)$')
            plt.plot(ref_pos_rec, color='black', linestyle='--', label='$x_{rm}(t)$')
            plt.plot(ref_rec, label='$r(t)$')
            plt.grid(True)
            plt.xlabel('time')
            plt.ylabel('Position $x(t)$')
            plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')

            ax2=plt.subplot(212)
            plt.plot(vel_rec, color='red', label='$\dot{x}(t)$')
            plt.plot(ref_vel_rec, color='black', linestyle='--', label='$\dot{x}_{rm}(t)$')
            plt.grid(True)
            plt.xlabel('time')
            plt.ylabel('Velocity $\dot{x}(t)$')
            plt.legend()

            return v,u