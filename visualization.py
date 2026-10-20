import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Button

from metrics import calculate_metrics


def run_animation(simulation):

    # =================================================
    # FIGURE LAYOUT
    # =================================================

    fig = plt.figure(
        figsize=(13, 8)
    )

    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            4.5,
            1.6
        ],
        wspace=0.12
    )

    # Left side:
    # main simulation
    ax = fig.add_subplot(
        grid[0, 0]
    )

    # Right side:
    # data panel
    info_ax = fig.add_subplot(
        grid[0, 1]
    )

    # Leave space for buttons
    plt.subplots_adjust(
        bottom=0.18
    )

    # Hide axes in information panel
    info_ax.axis(
        "off"
    )

    # =================================================
    # PARTICLES
    # =================================================

    scatter = ax.scatter(
        [],
        [],
        s=45,
        c=[],
        cmap="viridis",
        vmin=0,
        vmax=simulation.jam_density
    )

    ax.set_xlim(
        -1.3,
        simulation.width + 1.3
    )

    ax.set_ylim(
        -1.0,
        simulation.height + 1.0
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X Position (m)"
    )

    ax.set_ylabel(
        "Y Position (m)"
    )

    ax.set_title(
        "SPH Crowd Simulation"
    )

    # =================================================
    # DYNAMIC WALLS / EXITS
    # =================================================

    wall_artists = []
    exit_label_artists = []

    def clear_room_geometry():

        for artist in wall_artists:
            artist.remove()

        wall_artists.clear()

        for artist in exit_label_artists:
            artist.remove()

        exit_label_artists.clear()

    # =================================================
    # DRAW ONE WALL
    # =================================================

    def draw_wall_with_exits(
        side
    ):

        exits_on_side = [
            (
                index,
                exit_info
            )
            for index, exit_info
            in enumerate(
                simulation.exits
            )
            if (
                exit_info["side"]
                == side
            )
        ]

        exits_on_side.sort(
            key=lambda item:
            item[1]["start"]
        )

        if side in [
            "left",
            "right"
        ]:

            wall_length = (
                simulation.height
            )

        else:

            wall_length = (
                simulation.width
            )

        current = 0.0

        # -------------------------------------------------
        # Draw wall segments around exits
        # -------------------------------------------------

        for (
            index,
            exit_info
        ) in exits_on_side:

            start = (
                exit_info["start"]
            )

            end = (
                exit_info["end"]
            )

            # Solid section before exit
            if current < start:

                if side == "left":

                    line, = ax.plot(
                        [0, 0],
                        [current, start]
                    )

                elif side == "right":

                    line, = ax.plot(
                        [
                            simulation.width,
                            simulation.width
                        ],
                        [
                            current,
                            start
                        ]
                    )

                elif side == "bottom":

                    line, = ax.plot(
                        [
                            current,
                            start
                        ],
                        [0, 0]
                    )

                else:

                    line, = ax.plot(
                        [
                            current,
                            start
                        ],
                        [
                            simulation.height,
                            simulation.height
                        ]
                    )

                wall_artists.append(
                    line
                )

            # -------------------------------------------------
            # Exit label
            # -------------------------------------------------

            center = (
                start + end
            ) / 2.0

            if side == "right":

                text_artist = ax.text(
                    simulation.width + 0.15,
                    center,
                    f"EXIT {index + 1}",
                    va="center"
                )

            elif side == "left":

                text_artist = ax.text(
                    -1.15,
                    center,
                    f"EXIT {index + 1}",
                    va="center"
                )

            elif side == "top":

                text_artist = ax.text(
                    center,
                    simulation.height + 0.25,
                    f"EXIT {index + 1}",
                    ha="center"
                )

            else:

                text_artist = ax.text(
                    center,
                    -0.55,
                    f"EXIT {index + 1}",
                    ha="center"
                )

            exit_label_artists.append(
                text_artist
            )

            current = end

        # -------------------------------------------------
        # Final wall section
        # -------------------------------------------------

        if current < wall_length:

            if side == "left":

                line, = ax.plot(
                    [0, 0],
                    [
                        current,
                        wall_length
                    ]
                )

            elif side == "right":

                line, = ax.plot(
                    [
                        simulation.width,
                        simulation.width
                    ],
                    [
                        current,
                        wall_length
                    ]
                )

            elif side == "bottom":

                line, = ax.plot(
                    [
                        current,
                        wall_length
                    ],
                    [0, 0]
                )

            else:

                line, = ax.plot(
                    [
                        current,
                        wall_length
                    ],
                    [
                        simulation.height,
                        simulation.height
                    ]
                )

            wall_artists.append(
                line
            )

    # =================================================
    # DRAW FULL ROOM
    # =================================================

    def draw_room_geometry():

        clear_room_geometry()

        draw_wall_with_exits(
            "left"
        )

        draw_wall_with_exits(
            "right"
        )

        draw_wall_with_exits(
            "top"
        )

        draw_wall_with_exits(
            "bottom"
        )

    draw_room_geometry()

    # =================================================
    # RIGHT INFORMATION PANEL
    # =================================================

    title_text = info_ax.text(
        0.05,
        0.98,
        "SIMULATION DATA",
        fontsize=17,
        fontweight="bold",
        va="top"
    )

    # -------------------------------------------------
    # Remaining / Exited
    # -------------------------------------------------

    status_text = info_ax.text(
        0.05,
        0.86,
        "",
        fontsize=12,
        va="top"
    )

    # -------------------------------------------------
    # Exit counters
    # -------------------------------------------------

    exit_text = info_ax.text(
        0.05,
        0.72,
        "",
        fontsize=12,
        va="top"
    )

    # -------------------------------------------------
    # Playback speed
    # -------------------------------------------------

    animation_speed_text = info_ax.text(
        0.05,
        0.56,
        "",
        fontsize=12,
        va="top"
    )

    # -------------------------------------------------
    # Physical metrics
    # -------------------------------------------------

    metrics_text = info_ax.text(
        0.05,
        0.47,
        "",
        fontsize=11,
        va="top"
    )

    # =================================================
    # DENSITY COLORBAR SECTION
    # =================================================

    density_title = info_ax.text(
        0.05,
        0.16,
        "Crowd Density",
        fontsize=13,
        fontweight="bold",
        va="top"
    )

    colorbar_ax = (
        info_ax.inset_axes(
            [
                0.10,
                0.055,
                0.78,
                0.055
            ]
        )
    )

    colorbar = fig.colorbar(
        scatter,
        cax=colorbar_ax,
        orientation="horizontal"
    )

    colorbar.set_label(
        "Density (ped/m²)",
        fontsize=10
    )

    colorbar.ax.tick_params(
        labelsize=9
    )

    # =================================================
    # SIMULATION STATE
    # =================================================

    running = {
        "value": True
    }

    speed_multiplier = {
        "value": 1
    }

    # =================================================
    # UPDATE FUNCTION
    # =================================================

    def update(
        frame
    ):

        # -------------------------------------------------
        # Advance simulation
        # -------------------------------------------------

        if running["value"]:

            steps = (
                speed_multiplier[
                    "value"
                ]
            )

            for _ in range(
                steps
            ):

                simulation.step(
                    dt=0.01
                )

        # -------------------------------------------------
        # Update particle positions / density colors
        # -------------------------------------------------

        if (
            len(
                simulation.particles
            )
            > 0
        ):

            positions = np.array(
                [
                    particle.position
                    for particle
                    in simulation.particles
                ]
            )

            densities = np.array(
                [
                    particle.density
                    for particle
                    in simulation.particles
                ]
            )

            scatter.set_offsets(
                positions
            )

            scatter.set_array(
                densities
            )

        else:

            scatter.set_offsets(
                np.empty(
                    (0, 2)
                )
            )

            scatter.set_array(
                np.array([])
            )

        # =================================================
        # CALCULATE METRICS
        # =================================================

        metrics = calculate_metrics(
            simulation
        )

        # -------------------------------------------------
        # Remaining / Exited
        # -------------------------------------------------

        status_text.set_text(
            f"Remaining: "
            f"{metrics['remaining']}\n"
            f"Exited: "
            f"{metrics['exited']}"
        )

        # -------------------------------------------------
        # Exit counts
        # -------------------------------------------------

        exit_lines = []

        for (
            index,
            exit_info
        ) in enumerate(
            simulation.exits
        ):

            exit_lines.append(
                f"Exit {index + 1}: "
                f"{exit_info['count']}"
            )

        exit_text.set_text(
            "\n".join(
                exit_lines
            )
        )

        # -------------------------------------------------
        # Animation speed
        # -------------------------------------------------

        animation_speed_text.set_text(
            f"Animation Speed: "
            f"{speed_multiplier['value']}x"
        )

        # -------------------------------------------------
        # Physical values
        # -------------------------------------------------

        metrics_text.set_text(
            f"Average Speed: "
            f"{metrics['average_speed']:.2f} m/s\n"
            f"Maximum Speed: "
            f"{metrics['max_speed']:.2f} m/s\n\n"
            f"Average Density: "
            f"{metrics['average_density']:.2f} ped/m²\n"
            f"Maximum Density: "
            f"{metrics['max_density']:.2f} ped/m²"
        )

        return (
            scatter,
            status_text,
            exit_text,
            animation_speed_text,
            metrics_text
        )

    # =================================================
    # ANIMATION
    # =================================================

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=20,
        blit=False,
        cache_frame_data=False
    )

    # =================================================
    # BUTTONS
    # =================================================

    stop_ax = plt.axes(
        [
            0.16,
            0.05,
            0.17,
            0.06
        ]
    )

    speed_ax = plt.axes(
        [
            0.37,
            0.05,
            0.17,
            0.06
        ]
    )

    restart_ax = plt.axes(
        [
            0.58,
            0.05,
            0.17,
            0.06
        ]
    )

    stop_button = Button(
        stop_ax,
        "Stop"
    )

    speed_button = Button(
        speed_ax,
        "Speed Up"
    )

    restart_button = Button(
        restart_ax,
        "Restart"
    )

    # =================================================
    # STOP / RESUME
    # =================================================

    def stop_simulation(
        event
    ):

        running["value"] = (
            not running["value"]
        )

        if running["value"]:

            stop_button.label.set_text(
                "Stop"
            )

        else:

            stop_button.label.set_text(
                "Resume"
            )

    # =================================================
    # SPEED UP
    # =================================================

    def speed_up(
        event
    ):

        current_speed = (
            speed_multiplier[
                "value"
            ]
        )

        if current_speed == 1:

            speed_multiplier[
                "value"
            ] = 2

        elif current_speed == 2:

            speed_multiplier[
                "value"
            ] = 4

        elif current_speed == 4:

            speed_multiplier[
                "value"
            ] = 8

        else:

            speed_multiplier[
                "value"
            ] = 1

    # =================================================
    # RESTART
    # =================================================

    def restart_simulation(
        event
    ):

        # New random crowd
        # New random exits
        simulation.reset()

        running["value"] = True

        speed_multiplier[
            "value"
        ] = 1

        stop_button.label.set_text(
            "Stop"
        )

        # Exit locations changed,
        # redraw the walls
        draw_room_geometry()

        # Immediately display new particles
        positions = np.array(
            [
                particle.position
                for particle
                in simulation.particles
            ]
        )

        scatter.set_offsets(
            positions
        )

        # Density is not calculated until
        # next simulation step
        scatter.set_array(
            np.zeros(
                len(
                    simulation.particles
                )
            )
        )

        # Reset information panel immediately
        status_text.set_text(
            f"Remaining: "
            f"{len(simulation.particles)}\n"
            f"Exited: 0"
        )

        exit_text.set_text(
            "Exit 1: 0\n"
            "Exit 2: 0\n"
            "Exit 3: 0"
        )

        animation_speed_text.set_text(
            "Animation Speed: 1x"
        )

        metrics_text.set_text(
            "Average Speed: 0.00 m/s\n"
            "Maximum Speed: 0.00 m/s\n\n"
            "Average Density: 0.00 ped/m²\n"
            "Maximum Density: 0.00 ped/m²"
        )

        fig.canvas.draw_idle()

    # =================================================
    # CONNECT BUTTONS
    # =================================================

    stop_button.on_clicked(
        stop_simulation
    )

    speed_button.on_clicked(
        speed_up
    )

    restart_button.on_clicked(
        restart_simulation
    )

    # Keep animation reference alive
    fig._animation = ani

    plt.show()