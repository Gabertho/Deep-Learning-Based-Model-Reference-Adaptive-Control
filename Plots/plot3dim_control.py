import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3dim_control(qd_v, q_v, control):
    """
    Plota a trajetória desejada e real em 3D.

    Args:
    qd_v (np.array): Trajetória desejada.
    q_v (np.array): Trajetória real.
    control (int): Tipo de controle utilizado.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajetória desejada
    ax.plot3D(qd_v[0, :], qd_v[1, :], qd_v[2, :], 'b--', linewidth=2, label='Referência')

    # Verifica o tipo de controle para determinar a cor da linha
    if control == 1:  # Feedback Linearization
        cordalinha = 'g'
        label = 'PD (Feedback Linearization)'
    else:
        cordalinha = 'r'  # Pode ser ajustado para outros tipos de controle
        label = 'Outro Controle'

    # Trajetória real
    ax.plot3D(q_v[0, :], q_v[1, :], q_v[2, :], cordalinha, linewidth=2, label=label)

    # Configurações do gráfico
    ax.set_xlabel('$x(m)$', fontsize=17)
    ax.set_ylabel('$y(m)$', fontsize=17)
    ax.set_zlabel('$z(m)$', fontsize=17)
    ax.set_title('Gráfico trajeto real vs desejado')
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.show()

# Exemplo de uso:
# plot3dim_control(qd_v, q_v, control)
