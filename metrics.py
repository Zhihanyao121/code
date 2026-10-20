import numpy as np


def calculate_metrics(simulation):
    """
    Calculate real-time statistics from the current simulation.
    """

    # If everyone has already exited
    if len(simulation.particles) == 0:
        return {
            "average_speed": 0.0,
            "max_speed": 0.0,
            "min_speed": 0.0,

            "average_density": 0.0,
            "max_density": 0.0,
            "min_density": 0.0,

            "remaining": 0,
            "exited": len(
                simulation.finished_particles
            )
        }

    speeds = []
    densities = []

    for particle in simulation.particles:

        # Actual particle speed:
        # sqrt(vx^2 + vy^2)
        speed = np.linalg.norm(
            particle.velocity
        )

        speeds.append(
            speed
        )

        densities.append(
            particle.density
        )

    average_speed = np.mean(
        speeds
    )

    max_speed = np.max(
        speeds
    )

    min_speed = np.min(
        speeds
    )

    average_density = np.mean(
        densities
    )

    max_density = np.max(
        densities
    )

    min_density = np.min(
        densities
    )

    return {
        "average_speed": average_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,

        "average_density": average_density,
        "max_density": max_density,
        "min_density": min_density,

        "remaining": len(
            simulation.particles
        ),

        "exited": len(
            simulation.finished_particles
        )
    }