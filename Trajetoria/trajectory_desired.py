import numpy as np
from Trajetoria.traj_pol_f import traj_pol_f

from Trajetoria.trajectory_par_pol_f import trajectory_par_pol_f


def trajectory_desired(traj_des, q, dq, d2q, qf, dqf, d2qf, t, dt, traj_opt, vel_d, A, omg, to, tf, ax, ay, az, afi):
    """
    Planeja a trajetória para o drone.

    Args:
    traj_des (int): Tipo de trajetória desejada (1 para elipse, 2 para ponto a ponto).
    q (np.array): Estado atual.
    dq (np.array): Velocidade atual.
    d2q (np.array): Aceleração atual.
    qf (np.array): Estado final desejado.
    dqf (np.array): Velocidade final desejada.
    d2qf (np.array): Aceleração final desejada.
    t (float): Tempo atual.
    dt (float): Intervalo de tempo.
    traj_opt (int): Opção de trajetória (não utilizado nesta tradução).
    vel_d (float): Velocidade desejada.
    A (float): Amplitude da trajetória (para elipse).
    omg (float): Velocidade angular (para elipse).
    to (float): Tempo inicial (para trajetória ponto a ponto).
    tf (float): Tempo final (para trajetória ponto a ponto).
    ax, ay, az, afi (np.array): Parâmetros da trajetória (para ponto a ponto).

    Returns:
    tuple: Retorna a trajetória desejada e suas derivadas até a terceira ordem, além do tempo inicial e final atualizados.
    """
    if traj_des == 1:  # Trajetória de elipse
        xd = A * np.cos(omg * t)
        yd = A * np.sin(omg * t)
        zd = (A / 2) * np.sin(omg * t)
        fid = 0
        dxd = -omg * A * np.sin(omg * t)
        dyd = omg * A * np.cos(omg * t)
        dzd = (omg * A / 2) * np.cos(omg * t)
        dfid = 0
        d2xd = -(omg ** 2) * A * np.cos(omg * t)
        d2yd = -(omg ** 2) * A * np.sin(omg * t)
        d2zd = -(omg ** 2) * (A / 2) * np.sin(omg * t)
        d2fid = 0
        d3xd = d3yd = d3zd = d3fid = 0

    elif traj_des == 2:  # Trajetória ponto a ponto
        if (t / dt) == 1:
            to = t
            td = np.linalg.norm(qf - q) / vel_d
            tf = to + td
            ax, ay, az, afi = trajectory_par_pol_f(dt, to, tf, q, dq, d2q, qf, dqf, d2qf)

        if t <= tf:
            xd, dxd, d2xd, d3xd = traj_pol_f(ax, t, to)
            yd, dyd, d2yd, d3yd = traj_pol_f(ay, t, to)
            zd, dzd, d2zd, d3zd = traj_pol_f(az, t, to)
            fid, dfid, d2fid, d3fid = traj_pol_f(afi, t, to)

    else:
        raise ValueError("Trajetória inexistente")

    qd = np.array([xd, yd, zd, fid])
    dqd = np.array([dxd, dyd, dzd, dfid])
    d2qd = np.array([d2xd, d2yd, d2zd, d2fid])
    d3qd = np.array([d3xd, d3yd, d3zd, d3fid])

    return qd, dqd, d2qd, d3qd, to, tf
