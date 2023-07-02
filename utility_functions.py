# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:09:11 2023

@author: shaabang
"""
import numpy as np
import math


def skew_matrix(v):
    """
    Generate the skew-symmetric matrix corresponding to a 3D vector.

    Args:
        v (numpy.ndarray): Input 3D vector.

    Returns:
        numpy.ndarray: Skew-symmetric matrix representation of the input vector.
    """
    v = v.reshape(-1)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expSO3(xi):
    """
    Computes the exponential map on the special orthogonal group SO(3) given the Lie algebra xi.

    Args:
        xi (numpy.ndarray): Lie algebra element in R^3.

    Returns:
        numpy.ndarray: The corresponding rotation matrix in SO(3).

    """
    xi = xi.reshape(-1)
    theta = np.linalg.norm(xi)
    if theta == 0:
        return np.eye(3)

    skew_xi = skew_matrix(xi)

    # Compute the exponential map using the Baker-Campbell-Hausdorff formula.
    Rot = (
        np.eye(3)
        + np.sin(theta) / theta * skew_xi
        + 2 * np.sin(theta / 2) ** 2 / theta**2 * (skew_xi @ skew_xi)
    )

    return Rot


def rotation_matrix_to_euler(matrix):
    """
    Convert a 3x3 rotation matrix to Euler angles in XYZ order.
    The input rotation matrix  is from body to fixed frame (R from B to N)

    Args:
        matrix (numpy.ndarray): Input 3x3 rotation matrix.

    Returns:
        tuple: Euler angles in degrees, ordered as (X, Y, Z).
    """
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def euler_to_rotation_matrix(euler):
    """
    Convert Euler angles in XYZ order (in degrees) to a 3x3 rotation matrix.
    The resulting rotation matrix is from body to fixed frame (R from B to N)

    Args:
        euler (numpy.ndarray): Input Euler angles in degrees.

    Returns:
        numpy.ndarray: Output 3x3 rotation matrix.
    """

    euler = euler.reshape(-1)
    euler = np.pi / 180 * euler
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]

    Rz = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1],
        ]
    )

    Ry = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ]
    )

    return Rz @ Ry @ Rx


def RMSE_degrees(true, estimation):
    """
    Compute the root mean square error (RMSE) between true and estimated values given in degrees.

    Args:
        true (numpy.ndarray): Array of true values.
        estimation (numpy.ndarray): Array of estimated values.

    Returns:
        float: Root mean square error between true and estimated values.

    Notes:
        The input arrays should contain values in the range of -180 to 180.
    """
    diff = true - estimation
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)
    return np.sqrt(np.mean(np.square(diff)))
