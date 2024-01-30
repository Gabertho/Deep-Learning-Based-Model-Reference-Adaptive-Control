import numpy as np

from Controle.kernel import kernel

def gp_adaptativo_art(sigma_quad, l, zt_1, Zt, yt, i, omega_quad, etol, p_max):
    mu = np.zeros(4)
    cov = np.zeros((4, 4))

    for j in range(4):
        Ktt = np.zeros((len(Zt[j, :]), len(Zt[j, :])))
        kt = np.zeros((len(Zt[j, :]), len(Zt[j, :])))
        kt_s = np.zeros((len(Zt[j, :]), len(Zt[j, :])))
        
        for k in range(len(Zt[j, :])):
            Ktt[:, :] = kernel(sigma_quad, Zt[j, k], Zt, l, len(Zt[j, :]))
            kt[:, :] = kernel(sigma_quad, zt_1[j], Zt, l, len(Zt[j, :]))
            kt_s[:, :] = kernel(sigma_quad, zt_1[j], zt_1[j], l, len(zt_1[j]))

        Ct = np.diag(np.diag(Ktt)) + omega_quad * np.eye(i)
        Ct = np.linalg.cholesky(Ct)
        Bt = Ct @ yt
        mu[j] = Bt.T @ kt
        cov = kt_s - kt.T @ Ct @ kt

    return mu, cov, Zt, zt_1, yt
