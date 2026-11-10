import numpy as np

from kernel import (
    cubic_spline_kernel,
    cubic_spline_gradient
)


def compute_density(
    particles,
    h
):
    for particle_i in particles:

        density = 0.0

        for particle_j in particles:

            distance = np.linalg.norm(
                particle_i.position
                - particle_j.position
            )

            density += (
                particle_j.mass
                * cubic_spline_kernel(
                    distance,
                    h
                )
            )

        particle_i.density = density


def compute_pressure(
    particles,
    critical_density,
    pressure_coefficient
):
    for particle in particles:

        density_difference = (
            particle.density
            - critical_density
        )

        if density_difference > 0:

            particle.pressure = (
                pressure_coefficient
                * density_difference
            )

        else:
            particle.pressure = 0.0


def desired_speed_from_density(
    density,
    free_speed,
    critical_density,
    jam_density
):
    if density <= critical_density:
        return free_speed

    if density >= jam_density:
        return 0.0

    ratio = (
        jam_density - density
    ) / (
        jam_density
        - critical_density
    )

    return (
        free_speed * ratio
    )


# =================================================
# ARTIFICIAL VISCOSITY
# =================================================

def compute_artificial_viscosity(
    particle_i,
    particle_j,
    h,
    alpha=0.3,
    sound_speed=1.0,
    epsilon=0.01
):
    """
    Artificial viscosity term Pi_ij.

    This term is only active when two particles
    are moving toward each other.

    It helps reduce oscillation and numerical
    instability in high-density regions.
    """

    # Position difference
    r_ij = (
        particle_i.position
        - particle_j.position
    )

    # Velocity difference
    v_ij = (
        particle_i.velocity
        - particle_j.velocity
    )

    # Check whether particles
    # are approaching each other
    dot_product = np.dot(
        v_ij,
        r_ij
    )

    # If they are moving apart,
    # artificial viscosity is zero
    if dot_product >= 0:
        return 0.0

    distance_squared = np.dot(
        r_ij,
        r_ij
    )

    # mu_ij
    mu_ij = (
        h * dot_product
        / (
            distance_squared
            + epsilon * h ** 2
        )
    )

    average_density = (
        particle_i.density
        + particle_j.density
    ) / 2.0

    if average_density <= 1e-8:
        return 0.0

    # Pi_ij
    pi_ij = (
        -alpha
        * sound_speed
        * mu_ij
        / average_density
    )

    return pi_ij


# =================================================
# PRESSURE + VISCOSITY ACCELERATION
# =================================================

def compute_pressure_acceleration(
    particles,
    h
):
    pressure_accelerations = []

    epsilon = 1e-8

    for particle_i in particles:

        pressure_acceleration = (
            np.zeros(2)
        )

        for particle_j in particles:

            if particle_i is particle_j:
                continue

            distance = np.linalg.norm(
                particle_i.position
                - particle_j.position
            )

            # Cubic spline kernel
            # has support radius 2h
            if distance >= 2 * h:
                continue

            rho_i = max(
                particle_i.density,
                epsilon
            )

            rho_j = max(
                particle_j.density,
                epsilon
            )

            # --------------------------------
            # Artificial viscosity Pi_ij
            # --------------------------------

            viscosity = (
                compute_artificial_viscosity(
                    particle_i,
                    particle_j,
                    h
                )
            )

            # --------------------------------
            # Pressure + viscosity term
            # --------------------------------

            pressure_term = (
                particle_i.pressure
                / (rho_i ** 2)
                +
                particle_j.pressure
                / (rho_j ** 2)
                +
                viscosity
            )

            # Kernel gradient
            gradient = (
                cubic_spline_gradient(
                    particle_i.position,
                    particle_j.position,
                    h
                )
            )

            pressure_acceleration -= (
                particle_j.mass
                * pressure_term
                * gradient
            )

        pressure_accelerations.append(
            pressure_acceleration
        )

    return pressure_accelerations


# =================================================
# DRIVING / DESIRED MOTION
# =================================================

def compute_driving_acceleration(
    particle,
    target,
    free_speed,
    critical_density,
    jam_density,
    tau
):
    direction = (
        target
        - particle.position
    )

    distance = np.linalg.norm(
        direction
    )

    if distance > 0:

        direction = (
            direction
            / distance
        )

    else:

        direction = np.zeros(2)

    # Density-dependent desired speed
    desired_speed = (
        desired_speed_from_density(
            particle.density,
            free_speed,
            critical_density,
            jam_density
        )
    )

    particle.desired_speed = (
        desired_speed
    )

    desired_velocity = (
        desired_speed
        * direction
    )

    # Relaxation equation
    driving_acceleration = (
        desired_velocity
        - particle.velocity
    ) / tau

    return driving_acceleration


# =================================================
# TOTAL PARTICLE ACCELERATION
# =================================================

def compute_accelerations(
    particles,
    h,
    targets,
    free_speed,
    critical_density,
    jam_density,
    tau
):
    pressure_accelerations = (
        compute_pressure_acceleration(
            particles,
            h
        )
    )

    for index, particle in enumerate(
        particles
    ):

        target = targets[index]

        driving_acceleration = (
            compute_driving_acceleration(
                particle,
                target,
                free_speed,
                critical_density,
                jam_density,
                tau
            )
        )

        particle.acceleration = (
            driving_acceleration
            + pressure_accelerations[index]
        )