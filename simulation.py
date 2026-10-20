import numpy as np

from particle import Particle

from forces import (
    compute_density,
    compute_pressure,
    compute_accelerations
)


class CrowdSimulation:

    def __init__(
        self,
        width=12.0,
        height=8.0,
        number_of_particles=100
    ):

        self.width = width
        self.height = height

        self.number_of_particles = (
            number_of_particles
        )

        self.particles = []
        self.finished_particles = []

        self.statistics_printed = False

        # --------------------------------
        # SPH parameters
        # --------------------------------

        self.h = 1.25

        self.free_speed = 1.3

        self.critical_density = 1.35

        self.jam_density = 3.7

        self.tau = 0.5

        self.pressure_coefficient = 4.0

        self.max_speed = 2.0

        # --------------------------------
        # Random number generator
        # --------------------------------

        self.rng = np.random.default_rng()

        # --------------------------------
        # FIRST RUN:
        # all exits on right wall
        # --------------------------------

        self.exits = [
            {
                "side": "right",
                "start": 0.8,
                "end": 2.0,
                "count": 0
            },
            {
                "side": "right",
                "start": 3.4,
                "end": 4.6,
                "count": 0
            },
            {
                "side": "right",
                "start": 6.0,
                "end": 7.2,
                "count": 0
            }
        ]

        self.create_random_particles(
            self.number_of_particles
        )

    # =================================================
    # RANDOM PARTICLES
    # =================================================

    def create_random_particles(
        self,
        number_of_particles
    ):

        for _ in range(
            number_of_particles
        ):

            x = self.rng.uniform(
                0.5,
                self.width - 0.5
            )

            y = self.rng.uniform(
                0.5,
                self.height - 0.5
            )

            particle = Particle(
                position=[x, y],
                mass=1.0
            )

            self.particles.append(
                particle
            )

    # =================================================
    # RANDOM EXITS
    # =================================================

    def generate_random_exits(
        self,
        number_of_exits=3
    ):

        new_exits = []

        sides = [
            "left",
            "right",
            "top",
            "bottom"
        ]

        exit_width = 1.2
        corner_margin = 0.5

        max_attempts = 500
        attempts = 0

        while (
            len(new_exits)
            < number_of_exits
            and attempts < max_attempts
        ):

            attempts += 1

            side = self.rng.choice(
                sides
            )

            if side in [
                "left",
                "right"
            ]:

                wall_length = (
                    self.height
                )

            else:

                wall_length = (
                    self.width
                )

            minimum_start = (
                corner_margin
            )

            maximum_start = (
                wall_length
                - corner_margin
                - exit_width
            )

            if (
                maximum_start
                <= minimum_start
            ):
                continue

            start = self.rng.uniform(
                minimum_start,
                maximum_start
            )

            end = (
                start
                + exit_width
            )

            overlap = False

            for existing_exit in (
                new_exits
            ):

                if (
                    existing_exit["side"]
                    != side
                ):
                    continue

                existing_start = (
                    existing_exit["start"]
                )

                existing_end = (
                    existing_exit["end"]
                )

                gap = 0.3

                if not (
                    end + gap
                    < existing_start
                    or
                    start - gap
                    > existing_end
                ):

                    overlap = True
                    break

            if overlap:
                continue

            new_exits.append(
                {
                    "side": side,
                    "start": start,
                    "end": end,
                    "count": 0
                }
            )

        if (
            len(new_exits)
            < number_of_exits
        ):

            raise RuntimeError(
                "Could not generate "
                "non-overlapping exits."
            )

        self.exits = new_exits

    # =================================================
    # RESET / RESTART
    # =================================================

    def reset(
        self,
        number_of_particles=None
    ):

        if number_of_particles is None:

            number_of_particles = (
                self.number_of_particles
            )

        self.number_of_particles = (
            number_of_particles
        )

        self.particles = []
        self.finished_particles = []

        self.statistics_printed = False

        # Generate new exits
        self.generate_random_exits(
            number_of_exits=3
        )

        # Generate new crowd
        self.create_random_particles(
            number_of_particles
        )

        print()
        print("=" * 60)
        print(
            "SIMULATION RESTARTED"
        )
        print(
            "New random crowd + "
            "new random exits generated."
        )
        print("-" * 60)

        for index, exit_info in enumerate(
            self.exits
        ):

            print(
                f"Exit {index + 1}: "
                f"{exit_info['side']} wall "
                f"("
                f"{exit_info['start']:.2f}"
                f" - "
                f"{exit_info['end']:.2f}"
                f")"
            )

        print("=" * 60)
        print()

    # =================================================
    # EXIT CENTER
    # =================================================

    def get_exit_center(
        self,
        exit_info
    ):

        center = (
            exit_info["start"]
            + exit_info["end"]
        ) / 2.0

        side = (
            exit_info["side"]
        )

        if side == "right":

            return np.array(
                [
                    self.width + 1.0,
                    center
                ]
            )

        if side == "left":

            return np.array(
                [
                    -1.0,
                    center
                ]
            )

        if side == "top":

            return np.array(
                [
                    center,
                    self.height + 1.0
                ]
            )

        if side == "bottom":

            return np.array(
                [
                    center,
                    -1.0
                ]
            )

        raise ValueError(
            f"Unknown exit side: "
            f"{side}"
        )

    # =================================================
    # CHOOSE NEAREST EXIT
    # =================================================

    def choose_exit(
        self,
        particle
    ):

        best_exit_index = None
        best_target = None

        best_distance = float(
            "inf"
        )

        for index, exit_info in enumerate(
            self.exits
        ):

            target = (
                self.get_exit_center(
                    exit_info
                )
            )

            distance = np.linalg.norm(
                target
                - particle.position
            )

            if (
                distance
                < best_distance
            ):

                best_distance = (
                    distance
                )

                best_target = (
                    target
                )

                best_exit_index = (
                    index
                )

        return (
            best_exit_index,
            best_target
        )

    # =================================================
    # CHECK WHETHER POSITION IS INSIDE EXIT
    # =================================================

    def is_inside_exit(
        self,
        side,
        coordinate
    ):

        for exit_info in self.exits:

            if (
                exit_info["side"]
                != side
            ):

                continue

            if (
                exit_info["start"]
                <= coordinate
                <= exit_info["end"]
            ):

                return True

        return False

    # =================================================
    # WALL / BOUNDARY FORCE
    # =================================================

    def compute_wall_acceleration(
        self,
        particle
    ):

        wall_range = 0.6
        wall_strength = 4.0

        acceleration = np.zeros(2)

        x = particle.position[0]
        y = particle.position[1]

        # --------------------------------
        # LEFT WALL
        # --------------------------------

        if (
            x < wall_range
            and not self.is_inside_exit(
                "left",
                y
            )
        ):

            acceleration[0] += (
                wall_strength
                * (
                    1.0
                    - x / wall_range
                )
            )

        # --------------------------------
        # RIGHT WALL
        # --------------------------------

        right_distance = (
            self.width - x
        )

        if (
            right_distance < wall_range
            and not self.is_inside_exit(
                "right",
                y
            )
        ):

            acceleration[0] -= (
                wall_strength
                * (
                    1.0
                    - right_distance
                    / wall_range
                )
            )

        # --------------------------------
        # BOTTOM WALL
        # --------------------------------

        if (
            y < wall_range
            and not self.is_inside_exit(
                "bottom",
                x
            )
        ):

            acceleration[1] += (
                wall_strength
                * (
                    1.0
                    - y / wall_range
                )
            )

        # --------------------------------
        # TOP WALL
        # --------------------------------

        top_distance = (
            self.height - y
        )

        if (
            top_distance < wall_range
            and not self.is_inside_exit(
                "top",
                x
            )
        ):

            acceleration[1] -= (
                wall_strength
                * (
                    1.0
                    - top_distance
                    / wall_range
                )
            )

        return acceleration

    # =================================================
    # SIMULATION STEP
    # =================================================

    def step(
        self,
        dt
    ):

        if (
            len(self.particles)
            == 0
        ):

            if (
                not
                self.statistics_printed
            ):

                self.print_statistics()

                self.statistics_printed = (
                    True
                )

            return

        # --------------------------------
        # 1. Density
        # --------------------------------

        compute_density(
            self.particles,
            self.h
        )

        # --------------------------------
        # 2. Pressure
        # --------------------------------

        compute_pressure(
            self.particles,
            self.critical_density,
            self.pressure_coefficient
        )

        # --------------------------------
        # 3. Exit selection
        # --------------------------------

        targets = []

        for particle in self.particles:

            (
                exit_index,
                target
            ) = self.choose_exit(
                particle
            )

            particle.exit_index = (
                exit_index
            )

            targets.append(
                target
            )

        # --------------------------------
        # 4. Driving + pressure +
        #    artificial viscosity
        # --------------------------------

        compute_accelerations(
            self.particles,
            self.h,
            targets,
            self.free_speed,
            self.critical_density,
            self.jam_density,
            self.tau
        )

        # --------------------------------
        # 5. Wall / boundary force
        # --------------------------------

        for particle in self.particles:

            wall_acceleration = (
                self.compute_wall_acceleration(
                    particle
                )
            )

            particle.acceleration += (
                wall_acceleration
            )

        particles_to_remove = []

        # --------------------------------
        # 6. Euler integration
        # --------------------------------

        for particle in self.particles:

            particle.velocity += (
                particle.acceleration
                * dt
            )

            speed = np.linalg.norm(
                particle.velocity
            )

            if (
                speed
                > self.max_speed
            ):

                particle.velocity = (
                    particle.velocity
                    / speed
                    * self.max_speed
                )

            particle.position += (
                particle.velocity
                * dt
            )

            exit_index = (
                self.handle_boundaries(
                    particle
                )
            )

            if (
                exit_index
                is not None
            ):

                particles_to_remove.append(
                    (
                        particle,
                        exit_index
                    )
                )

        # --------------------------------
        # 7. Remove exited particles
        # --------------------------------

        for (
            particle,
            exit_index
        ) in particles_to_remove:

            self.particles.remove(
                particle
            )

            self.finished_particles.append(
                particle
            )

            self.exits[
                exit_index
            ]["count"] += 1

    # =================================================
    # FIND EXIT ON WALL
    # =================================================

    def find_exit_on_wall(
        self,
        side,
        coordinate
    ):

        for index, exit_info in enumerate(
            self.exits
        ):

            if (
                exit_info["side"]
                != side
            ):

                continue

            if (
                exit_info["start"]
                <= coordinate
                <= exit_info["end"]
            ):

                return index

        return None

    # =================================================
    # HARD BOUNDARY SAFETY CHECK
    # =================================================

    def handle_boundaries(
        self,
        particle
    ):

        margin = 0.15

        x = particle.position[0]
        y = particle.position[1]

        # --------------------------------
        # LEFT WALL
        # --------------------------------

        if x <= margin:

            exit_index = (
                self.find_exit_on_wall(
                    "left",
                    y
                )
            )

            if exit_index is not None:

                if x < 0:
                    return exit_index

            else:

                particle.position[0] = (
                    margin
                )

                particle.velocity[0] = (
                    abs(
                        particle.velocity[0]
                    )
                    * 0.2
                )

        # --------------------------------
        # RIGHT WALL
        # --------------------------------

        if (
            x
            >= self.width - margin
        ):

            exit_index = (
                self.find_exit_on_wall(
                    "right",
                    y
                )
            )

            if exit_index is not None:

                if x > self.width:
                    return exit_index

            else:

                particle.position[0] = (
                    self.width - margin
                )

                particle.velocity[0] = (
                    -abs(
                        particle.velocity[0]
                    )
                    * 0.2
                )

        # --------------------------------
        # BOTTOM WALL
        # --------------------------------

        if y <= margin:

            exit_index = (
                self.find_exit_on_wall(
                    "bottom",
                    x
                )
            )

            if exit_index is not None:

                if y < 0:
                    return exit_index

            else:

                particle.position[1] = (
                    margin
                )

                particle.velocity[1] = (
                    abs(
                        particle.velocity[1]
                    )
                    * 0.2
                )

        # --------------------------------
        # TOP WALL
        # --------------------------------

        if (
            y
            >= self.height - margin
        ):

            exit_index = (
                self.find_exit_on_wall(
                    "top",
                    x
                )
            )

            if exit_index is not None:

                if y > self.height:
                    return exit_index

            else:

                particle.position[1] = (
                    self.height - margin
                )

                particle.velocity[1] = (
                    -abs(
                        particle.velocity[1]
                    )
                    * 0.2
                )

        return None

    # =================================================
    # FINAL STATISTICS
    # =================================================

    def print_statistics(
        self
    ):

        print()
        print("=" * 60)
        print(
            "SIMULATION FINISHED"
        )
        print("=" * 60)

        total_exited = len(
            self.finished_particles
        )

        print(
            f"Total pedestrians exited: "
            f"{total_exited}"
        )

        print("-" * 60)

        for index, exit_info in enumerate(
            self.exits
        ):

            print(
                f"Exit {index + 1}: "
                f"{exit_info['count']} "
                f"pedestrians "
                f"| "
                f"{exit_info['side']} wall"
            )

        print("-" * 60)

        if total_exited > 0:

            for index, exit_info in enumerate(
                self.exits
            ):

                percentage = (
                    exit_info["count"]
                    / total_exited
                    * 100
                )

                print(
                    f"Exit {index + 1} usage: "
                    f"{percentage:.1f}%"
                )

        print("=" * 60)
        print()