def traj_pol_f(a, t, to):
    """
    Calcula a trajetória polinomial e suas derivadas.

    Args:
    a (np.array): Coeficientes do polinômio.
    t (float): Tempo atual.
    to (float): Tempo inicial.

    Returns:
    tuple: Posição (q), velocidade (dq), aceleração (d2q) e aceleração angular (d3q) no tempo t.
    """
    ao, a1, a2, a3, a4, a5 = a
    delta_t = t - to

    q = ao + a1 * delta_t + a2 * delta_t**2 + a3 * delta_t**3 + a4 * delta_t**4 + a5 * delta_t**5
    dq = a1 + 2 * a2 * delta_t + 3 * a3 * delta_t**2 + 4 * a4 * delta_t**3 + 5 * a5 * delta_t**4
    d2q = 2 * a2 + 6 * a3 * delta_t + 12 * a4 * delta_t**2 + 20 * a5 * delta_t**3
    d3q = 6 * a3 + 24 * a4 * delta_t + 60 * a5 * delta_t**2

    return q, dq, d2q, d3q

