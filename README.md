# IC
# Intelligent Adaptive Control of a Quadcopter

This is a research project funded by the São Paulo Research Foundation (FAPESP) and conducted at the Laboratory of Autonomous Robots and Intelligent Systems (LARIS), in the Department of Computer Science (DC) at the Federal University of São Carlos (Brazil).

## Overview

Unmanned Aerial Vehicles (UAVs), also known as Drones, often operate under adverse environmental conditions where external disturbances such as wind gusts and parametric uncertainties in the dynamic model of these vehicles - like mass, center of gravity, and moments of inertia - are constantly present. In this scenario, it is crucial to ensure that the control system used is robust enough to handle such disturbances, ensuring the safety, stability, and efficiency of the aircraft.

In this Scientific Initiation work, we have chosen to develop an intelligent model reference adaptive control to ensure the robustness of the system of a quadcopter drone, considering the aforementioned scenario and the tracking of its trajectory. To achieve this, deep neural networks will be employed to learn the parametric uncertainties and estimate the external disturbances of this aircraft. The implementation of the control architecture will be developed using Python and the TensorFlow framework.

For experimental validation, we intend to implement the simulated architecture using the "bebop_autonomy" driver for the Robot Operating System (ROS) framework on a commercial quadcopter, the Parrot Bebop 2, which already has stabilization control. Finally, the final validation of the system will be conducted using the state-of-the-art drone simulation tool, Parrot Sphinx.


## Contact

For more information about this project, please contact the LARIS team at the Federal University of São Carlos:

- Email: [gabrielbertho@estudante.ufscar.br](gabrielbertho@estudante.ufscar.br)

## Further information and references
https://github.com/girishvjoshi/DeepMRAC
https://ieeexplore.ieee.org/abstract/document/9029173
https://repositorio.ufscar.br/handle/ufscar/17785?show=full
