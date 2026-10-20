from simulation import (
    CrowdSimulation
)

from visualization import (
    run_animation
)


def main():

    simulation = (
        CrowdSimulation(
            width=12.0,
            height=8.0,
            number_of_particles=100
        )
    )

    run_animation(
        simulation
    )


if __name__ == "__main__":
    main()