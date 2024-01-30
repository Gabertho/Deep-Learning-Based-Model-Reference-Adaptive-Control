import numpy as np
from Trajetoria.traj_pol_f import traj_pol_f

from Trajetoria.trajectory_par_pol_f import trajectory_par_pol_f


def trajectory_desired(traj_des, q, dq, d2q, qf, dqf, d2qf, t, dt, traj_opt, vel_d, A, omg, to, tf, ax, ay, az, afi):
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
        d3xd = 0
        d3yd = 0
        d3zd = 0
        d3fid = 0

    elif traj_des == 2:  # Trajetória ponto final
        if t / dt == 1:
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
        print('trajetória inexistente')

    qd = np.array([xd, yd, zd, fid])
    dqd = np.array([dxd, dyd, dzd, dfid])
    d2qd = np.array([d2xd, d2yd, d2zd, d2fid])
    d3qd = np.array([d3xd, d3yd, d3zd, d3fid])

    return qd, dqd, d2qd, d3qd, to, tf, ax, ay, az, afi
