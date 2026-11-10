import math
import numpy as np


def cubic_spline_kernel(distance, h):
    q = distance / h

    sigma = 10.0 / (
        7.0 * math.pi * h ** 2
    )

    if 0 <= q < 1:
        return sigma * (
            1.0
            - 1.5 * q ** 2
            + 0.75 * q ** 3
        )

    elif 1 <= q < 2:
        return sigma * (
            0.25 * (2.0 - q) ** 3
        )

    return 0.0


def cubic_spline_gradient(
    position_i,
    position_j,
    h
):
    difference = (
        position_i
        - position_j
    )

    distance = np.linalg.norm(
        difference
    )

    if distance == 0:
        return np.zeros(2)

    q = distance / h

    sigma = 10.0 / (
        7.0 * math.pi * h ** 2
    )

    if 0 <= q < 1:

        dW_dq = sigma * (
            -3.0 * q
            + 2.25 * q ** 2
        )

    elif 1 <= q < 2:

        dW_dq = (
            -0.75
            * sigma
            * (2.0 - q) ** 2
        )

    else:
        return np.zeros(2)

    dW_dr = (
        dW_dq / h
    )

    direction = (
        difference / distance
    )

    return (
        dW_dr * direction
    )