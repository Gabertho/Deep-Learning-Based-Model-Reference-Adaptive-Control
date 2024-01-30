import numpy as np
from Controle.kernel import kernel
from scipy.linalg import cholesky

# Ajuste na função GPoff para usar a função de kernel corrigida
def GPoff(sigma_quad, l, Ytrain, Xtrain, Xtest):
    # Ytrain deve ser (4, 2000)
    # Xtrain e Xtest devem ser (2000, 4), cada coluna representa uma amostra para uma variável
    M = Ytrain.shape[0]  # Número de variáveis, que é 4
    n = Xtrain.shape[0]  # Número de amostras de treinamento, que é 2000

    # Matriz de kernel entre os dados de treino (K_xx) e entre os dados de treino e de teste (K_sx)
    K_xx = kernel(sigma_quad, Xtrain, Xtrain, l)
    K_sx = kernel(sigma_quad, Xtrain, Xtest, l)
    
    # L é a decomposição de Cholesky de K_xx
    L = np.linalg.cholesky(K_xx + sigma_quad * np.eye(n))

    # Resolva para cada dimensão do output
    mu = np.zeros((M, Xtest.shape[0]))
    cov = np.zeros((Xtest.shape[0], Xtest.shape[0], M))

    for i in range(M):
        # Ytrain[i, :] é um vetor (1, n)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ytrain[i, :]))
        mu[i, :] = K_sx @ alpha
        v = np.linalg.solve(L, K_sx.T)
        cov[:, :, i] = K_sx.T @ K_sx - v.T @ v

    return mu, cov