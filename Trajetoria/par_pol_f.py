import numpy as np

def par_pol_f(to, tf, qo, dqo, d2qo, qf, dqf, d2qf):
    # Verifique se to e tf são diferentes para evitar uma matriz singular
    if np.isclose(to, tf):
        print(to, tf)
        raise ValueError("Os tempos inicial e final são muito próximos ou iguais, o que pode causar uma matriz singular.")

    # Defina os vetores e a matriz
    q_v = [qo, dqo, d2qo, qf, dqf, d2qf]
    delta_t = tf - to

    A = [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 2, 0, 0, 0],
         [1, delta_t, delta_t**2, delta_t**3, delta_t**4, delta_t**5],
         [0, 1, 2*delta_t, 3*delta_t**2, 4*delta_t**3, 5*delta_t**4],
         [0, 0, 2, 6*delta_t, 12*delta_t**2, 20*delta_t**3]]

    # Use a pseudo-inversa para calcular os coeficientes
    try:
        a = np.linalg.pinv(A) @ q_v
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("A matriz A é singular e não pode ser invertida.")

    return a
