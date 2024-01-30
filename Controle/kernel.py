import numpy as np

def kernel(sigma_quad, x1, x2, l):
    # Inicializa a matriz de kernel com zeros
    k = np.zeros((x2.shape[0], x1.shape[0]))

    # Calcula o kernel gaussiano para cada par de pontos
    for i in range(x2.shape[0]):
        diff = x1 - x2[i, :]
        k[i, :] = sigma_quad * np.exp(-np.sum(diff**2, axis=1) / (2 * l**2))
    
    return k