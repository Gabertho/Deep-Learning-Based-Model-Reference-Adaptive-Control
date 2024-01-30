
from Trajetoria.par_pol_f import par_pol_f


def trajectory_par_pol_f(dt, to, tf, so, dso, d2so, sf, dsf, d2sf):
    """
    Calcula os parâmetros de uma trajetória polinomial para cada dimensão.

    Args:
    dt (float): Intervalo de tempo.
    to (float): Tempo inicial.
    tf (float): Tempo final.
    so (np.array): Estado inicial [x, y, z, orientação].
    dso (np.array): Velocidade inicial [vx, vy, vz, v_orientação].
    d2so (np.array): Aceleração inicial [ax, ay, az, a_orientação].
    sf (np.array): Estado final [x, y, z, orientação].
    dsf (np.array): Velocidade final [vx, vy, vz, v_orientação].
    d2sf (np.array): Aceleração final [ax, ay, az, a_orientação].

    Returns:
    tuple: Coeficientes dos polinômios para cada dimensão da trajetória.
    """
    xo, yo, zo, fio = so
    dxo, dyo, dzo, dfio = dso
    d2xo, d2yo, d2zo, d2fio = d2so
    xf, yf, zf, fif = sf
    dxf, dyf, dzf, dfif = dsf
    d2xf, d2yf, d2zf, d2fif = d2sf

    ax = par_pol_f(to, tf, xo, dxo, d2xo, xf, dxf, d2xf)
    ay = par_pol_f(to, tf, yo, dyo, d2yo, yf, dyf, d2yf)
    az = par_pol_f(to, tf, zo, dzo, d2zo, zf, dzf, d2zf)
    afi = par_pol_f(to, tf, fio, dfio, d2fio, fif, dfif, d2fif)

    return ax, ay, az, afi

