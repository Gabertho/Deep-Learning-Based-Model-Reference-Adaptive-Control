import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3dim_control(qd_v, q_v, control):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajetória desejada
    ax.plot(qd_v[0, :], qd_v[1, :], qd_v[2, :], 'b--', linewidth=2, label='Referência')

    # Nome dos tipos de controle
    control_names = {
        1: 'FL',
        2: 'H∞',
        3: 'H∞ RN',
        4: 'H∞ GP THETA',
        5: 'H∞ GP OFF',
    }

    # Cor da linha baseada no tipo de controle
    color_map = {
        1: 'g',  # FL
        2: 'r',  # H∞
        3: 'r',  # H∞ RN
        4: 'r',  # H∞ GP THETA
        5: 'r',  # H∞ GP OFF
    }
    cordalinha = color_map.get(control, 'k')  # Padrão para cor preta se o controle não for reconhecido

    # Trajetória real
    control_label = control_names.get(control, 'Desconhecido')
    ax.plot(q_v[0, :], q_v[1, :], q_v[2, :], cordalinha, linewidth=2, label=control_label)

    # Configurações do gráfico
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.title('Gráfico trajeto real vs desejado')

    plt.show()

# Exemplo de uso
# plot3dim_control(qd_v, q_v, control)
