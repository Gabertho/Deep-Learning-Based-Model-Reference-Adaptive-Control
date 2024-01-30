import numpy as np

def normalize_angle_f(r, low):
    """
    Normaliza um ângulo para um intervalo semi-fechado [low, low + 2π).

    Args:
    r (float): Ângulo em radianos.
    low (float): Valor inicial do intervalo de normalização.

    Returns:
    float: Ângulo equivalente no intervalo [low, low + 2π).
    """
    return r - 2 * np.pi * np.floor((r - low) / (2 * np.pi))
