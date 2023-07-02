# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:59:04 2023

@author: Ghadeer SHAABAN
Ph.D. student at GIPSA-Lab
University Grenoble Alpes, France


This code offers a straightforward implementation of the Invariant Extended 
Kalman Filter for attitude estimation (Kalman filter on SO(3)).
The estimation is performed based on observations of two vectors with known 
values in the fixed frame.
"""


# import Libraries
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import (
    expSO3,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    RMSE_degrees,
    skew_matrix,
)


"""
##############################################################################
###################       Dynamic and output functions     ###################      
##############################################################################
"""


def f(Rot, w_i, dt):
    """
    Update the rotation matrix based on angular velocity.

    Args:
        Rot (numpy.ndarray): Rotation matrix from B to N (R_B^N).
        w_i (numpy.ndarray): Angular velocity.
        dt (float): Time step.

    Returns:
        numpy.ndarray: Updated rotation matrix (R_next_B^N).
    """
    Rot_next = Rot @ expSO3(w_i * dt)
    return Rot_next


def h(Rot, V1N):
    """
    Compute the transformed vector V1b based on the rotation matrix and a given vector V1N.

    Args:
        Rot (numpy.ndarray): Rotation matrix from B to N (R_B^N).
        V1N (numpy.ndarray): Input vector V1N in the navigation (fixed) frame.

    Returns:
        numpy.ndarray: Transformed vector V1B in the body frame.
    """
    V1B = Rot.T @ V1N.reshape(-1, 1)
    return V1B


"""
##############################################################################
###################        Generate True data              ###################      
##############################################################################
"""

N = 10000  # number of samples
total_time = 100  # time duration
timeVector = np.linspace(0, total_time, N)
dt = total_time / N


# true angular velocity
w_true = np.zeros((N, 3))  # [wx ; wy; wz]

for i in range(N):
    w_true[i, 0] = 0.8 * np.cos(1.2 * timeVector[i])  # wx
    w_true[i, 1] = -1.1 * np.cos(0.5 * timeVector[i])  # wy
    w_true[i, 2] = -0.4 * np.cos(0.3 * timeVector[i])  # wz


# rotaton matrix, from B to N (R_B^N)
Rot_true = np.zeros((3, 3, N))
# start the true value of rotation corresponding to (45,45,45) degrees euler angles
Rot_true[:, :, 0] = euler_to_rotation_matrix(np.array([45, 45, 45]))

for i in range(1, N):
    Rot_true[:, :, i] = f(Rot_true[:, :, i - 1], w_true[i - 1, :], dt)

euler_true = np.zeros((N, 3))
for i in range(N):
    euler_true[i, :] = np.array(rotation_matrix_to_euler(Rot_true[:, :, i]))


# V1N, V2N could be any two known vectors in fixed (navigation) frame,
# example here, the measured gravity and magnitic field,
V1N_true = np.array([0, 0, 9.81])  # m.sec^-2
V2N_true = np.array([0.23, 0.01, 0.41])  # Gauss


V1B_true = np.zeros((N, 3))
V2B_true = np.zeros((N, 3))
for i in range(N):
    # Rot_true:  from B to N
    # Rot_true.T : from N to B (R_N^B)
    V1B_true[i, :] = Rot_true[:, :, i].T @ V1N_true
    V2B_true[i, :] = Rot_true[:, :, i].T @ V2N_true


"""
##############################################################################
##################     Generate noisy measured vector       ##################      
##############################################################################
"""

sigma_v1 = (
    0.01  # (for our example, it the the noise of accelerometer m.sec ^-2)
)
sigma_v2 = 0.005  # (for our example, it the the noise of magnitometer Gauss)
sigma_w = 0.01  # rad.sec ^-2


V1B_meas = V1B_true + np.random.normal(0, sigma_v1, (N, 3))
V2B_meas = V2B_true + np.random.normal(0, sigma_v2, (N, 3))
w_meas = w_true + np.random.normal(0, sigma_w, (N, 3))


"""
##############################################################################
###################             IEKF Algorithm             ###################      
##############################################################################
"""


def IEKF(Rot, w_i, P, R, Q, V1N, V1B, V2N, V2B):
    """
    Perform the Invariant Extended Kalman Filter (IEKF) for attitude estimation.

    Args:
        Rot (numpy.ndarray): Current rotation matrix from B to N (R_B^N).
        w_i (numpy.ndarray): Angular velocity.
        P (numpy.ndarray): Estimation error covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        Q (numpy.ndarray): Process noise covariance matrix.
        V1N (numpy.ndarray): Measurement vector V1N in the navigation (fixed) frame.
        V1B (numpy.ndarray): Measurement vector V1B in the body frame.
        V2N (numpy.ndarray): Measurement vector V2N in the navigation (fixed) frame.
        V2B (numpy.ndarray): Measurement vector V2B in the body frame.

    Returns:
        tuple: Updated rotation matrix (R_next_B^N) and covariance matrix.
    """
    Rot_p = f(Rot, w_i, dt)
    A = np.eye(3)
    E = np.eye(3)
    Pp = A @ P @ A.T + E @ Q @ E.T

    H = np.vstack((skew_matrix(Rot_p.T @ V1N), skew_matrix(Rot_p.T @ V2N)))

    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    P = Pp - K @ H @ Pp

    innovation = np.vstack(
        (
            V1B.reshape(-1, 1) - h(Rot_p, V1N),
            V2B.reshape(-1, 1) - h(Rot_p, V2N),
        )
    )
    xi = K @ innovation
    Rot_next = Rot_p @ expSO3(xi.reshape(-1))

    return Rot_next, P


# Measurement noise covariance matrix.
V1_var = sigma_v1**2
V2_var = sigma_v2**2
R_meas = np.diag([V1_var, V1_var, V1_var, V2_var, V2_var, V2_var])
# Process noise covariance matrix.
w_var = (dt * sigma_w) ** 2
Q = np.diag([w_var, w_var, w_var])
# initial estimation
Rot0 = np.eye(3)
# initial estimation error covariance matrix
P0 = np.eye(3)
P_IEKF = np.copy(P0)
# sequence of rotation matrices estimations.
IEKF_estimation = np.zeros((3, 3, N))
IEKF_estimation[:, :, 0] = np.copy(Rot0)
Rot_estimation = np.copy(Rot0)


for i in range(N - 1):
    [Rot_estimation, P_IEKF] = IEKF(
        Rot_estimation,
        w_meas[i],
        P_IEKF,
        R_meas,
        Q,
        V1N_true,
        V1B_meas[i],
        V2N_true,
        V2B_meas[i],
    )
    IEKF_estimation[:, :, i + 1] = np.copy(Rot_estimation)

"""
##############################################################################
###################             Results and plots          ###################      
##############################################################################
"""

euler_estimated_IEKF = np.zeros((N, 3))
for i in range(N):
    euler_estimated_IEKF[i, :] = np.array(
        rotation_matrix_to_euler(IEKF_estimation[:, :, i])
    )


start_index = 1000
RMSE_euler_IEKF_Degree = RMSE_degrees(
    euler_true[start_index:, :], euler_estimated_IEKF[start_index:, :]
)
print("RMSE_euler_IEKF_Degree:", RMSE_euler_IEKF_Degree)


eulerIndex = 0  # 0 roll, 1 pitch, 2 yaw
titles = ["roll estimation", "pitch estimation", "yaw estimation"]
for eulerIndex in range(3):
    plt.figure(eulerIndex)
    plt.plot(timeVector, euler_estimated_IEKF[:, eulerIndex])
    plt.plot(timeVector, euler_true[:, eulerIndex])
    plt.legend(["IEKF", "true"])
    plt.title(titles[eulerIndex])
    plt.show
