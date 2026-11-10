"""Single-window SPH crowd viewer with queue/stuck status."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.colors import Normalize


def draw_room(ax, sim):
    ax.clear()
    for side in ('left', 'right', 'bottom', 'top'):
        openings = sorted((e for e in sim.exits if e['side'] == side), key=lambda e: e['start'])
        wall_length = sim.height if side in ('left', 'right') else sim.width
        pieces = []
        prev = 0.0
        for e in openings:
            pieces.append((prev, e['start'], 'black', 2))
            pieces.append((e['start'], e['end'], 'green', 5))
            prev = e['end']
        pieces.append((prev, wall_length, 'black', 2))
        for a, b, color, thick in pieces:
            if b <= a:
                continue
            if side == 'left':
                xs, ys = (0, 0), (a, b)
            elif side == 'right':
                xs, ys = (sim.width, sim.width), (a, b)
            elif side == 'bottom':
                xs, ys = (a, b), (0, 0)
            else:
                xs, ys = (a, b), (sim.height, sim.height)
            ax.plot(xs, ys, color=color, linewidth=thick, solid_capstyle='butt')
    ax.set_xlim(-0.7, sim.width + 0.7)
    ax.set_ylim(-0.5, sim.height + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')


def run_single_viewer(simulation, experiment_mode=False,
                      number_of_particles=100, random_seed=42,
                      exit_widths=(0.6, 1.2, 2.4)):
    from simulation import CrowdSimulation

    state = {'simulation': simulation, 'index': 0, 'running': True, 'speed': 1}
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.84, bottom=0.31)
    color_norm = Normalize(vmin=0, vmax=1.5)  # Display scale only; no physics change
    plot = {'scatter': None}

    info = fig.text(0.5, 0.25, '', ha='center', va='center', fontsize=11)
    queue_status = fig.text(0.5, 0.205, '', ha='center', va='center', fontsize=12,
                            fontweight='bold')
    queue_details = fig.text(0.5, 0.163, '', ha='center', va='center', fontsize=10)

    def redraw_scene():
        sim = state['simulation']
        draw_room(ax, sim)
        if experiment_mode:
            ax.set_title(f'Exit Width: {exit_widths[state["index"]]:.1f} m')
            fig.suptitle('Experiment 1: Exit Width and Evacuation', fontsize=15)
        else:
            ax.set_title('SPH Crowd Simulation')
            fig.suptitle('SPH Crowd Simulation', fontsize=15)
        plot['scatter'] = ax.scatter([], [], s=45, c=[], cmap='viridis', norm=color_norm)
        refresh_particles()

    def refresh_particles():
        sim = state['simulation']
        if sim.particles:
            positions = np.asarray([p.position for p in sim.particles])
            densities = np.asarray([p.density for p in sim.particles])
        else:
            positions = np.empty((0, 2))
            densities = np.array([])
        plot['scatter'].set_offsets(positions)
        plot['scatter'].set_array(densities)

    def refresh_status():
        sim = state['simulation']
        remaining = len(sim.particles)
        exited = len(sim.finished_particles)
        densities = [p.density for p in sim.particles]
        max_density = max(densities, default=0.0)
        info.set_text(
            f'Remaining: {remaining}  |  Exited: {exited}  |  Time: {sim.elapsed_time:.2f}s'
            f'  |  Max SPH Density: {max_density:.2f}'
        )
        stuck_now = sum(bool(getattr(p, 'stuck', False)) for p in sim.particles)
        q = sim.queue_diagnostics() if hasattr(sim, 'queue_diagnostics') else {}
        active = bool(q.get('queue_active', False))
        queue_status.set_text(
            f'STUCK: {stuck_now} pedestrian(s)  |  '
            f'QUEUE: {"ACTIVE" if active else "NORMAL"}'
        )
        queue_status.set_color('#b45309' if active or stuck_now else 'green')
        groups = q.get('per_exit', [])
        description = []
        for idx, group in enumerate(groups):
            description.append(
                f'Exit {idx+1}: {"QUEUE ON" if group.get("active") else "Normal"}'
                f' | Nearby {group.get("nearby", 0)}'
                f' | Stuck {group.get("stuck", 0)}'
                f' | Avg speed {group.get("mean_speed", 0.0):.2f} m/s'
            )
        queue_details.set_text(
            '    '.join(description) +
            f'\nQueue activations: {q.get("queue_activations", 0)}'
        )

    def create_sim(index):
        return CrowdSimulation(
            width=simulation.width,
            height=simulation.height,
            number_of_particles=number_of_particles,
            exit_width=exit_widths[index],
            random_seed=random_seed,
            radius=simulation.radius,
        )

    redraw_scene()
    refresh_status()
    button_specs = [
        (0.14, 'Pause'), (0.33, 'Speed: 1x'),
        (0.52, 'Restart'), (0.71, 'Next')
    ]
    buttons = [Button(fig.add_axes([x, 0.045, 0.15, 0.055]), title)
               for x, title in button_specs]
    pause_button, speed_button, restart_button, next_button = buttons
    if not experiment_mode:
        next_button.set_active(False)

    def reset_controls():
        state['running'] = True
        state['speed'] = 1
        pause_button.label.set_text('Pause')
        speed_button.label.set_text('Speed: 1x')

    def pause(_):
        state['running'] = not state['running']
        pause_button.label.set_text('Pause' if state['running'] else 'Resume')
        fig.canvas.draw_idle()

    def speed(_):
        state['speed'] = {1: 2, 2: 4, 4: 1}[state['speed']]
        speed_button.label.set_text(f'Speed: {state["speed"]}x')
        fig.canvas.draw_idle()

    def restart(_):
        if experiment_mode:
            state['simulation'] = create_sim(state['index'])
        else:
            state['simulation'].reset()
        reset_controls()
        redraw_scene()
        refresh_status()
        fig.canvas.draw_idle()

    def next_scene(_):
        if not experiment_mode:
            return
        state['index'] = (state['index'] + 1) % len(exit_widths)
        state['simulation'] = create_sim(state['index'])
        reset_controls()
        redraw_scene()
        refresh_status()
        fig.canvas.draw_idle()

    for button, callback in zip(buttons, [pause, speed, restart, next_scene]):
        button.on_clicked(callback)

    def update(_):
        sim = state['simulation']
        if state['running'] and sim.particles:
            for _ in range(state['speed']):
                if not sim.particles:
                    break
                sim.step(0.02)
        refresh_particles()
        refresh_status()
        return (plot['scatter'],)

    ani = FuncAnimation(fig, update, interval=50, blit=False,
                        cache_frame_data=False)
    fig._animation = ani
    fig._buttons = buttons
    plt.show()


def run_animation(simulation):
    run_single_viewer(simulation)


def run_experiment_comparison(number_of_particles=100, random_seed=42,
                              exit_widths=(0.6, 1.2, 2.4)):
    from simulation import CrowdSimulation
    first = CrowdSimulation(width=12.0, height=3.6,
                            number_of_particles=number_of_particles,
                            exit_width=exit_widths[0],
                            random_seed=random_seed, radius=0.22)
    run_single_viewer(first, True, number_of_particles, random_seed, exit_widths)


if __name__ == '__main__':
    from simulation import CrowdSimulation
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store_true')
    parser.add_argument('--people', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.people < 1:
        parser.error('--people must be positive')
    if args.experiment:
        run_experiment_comparison(args.people, args.seed)
    else:
        run_animation(CrowdSimulation(width=12.0, height=3.6,
                                      number_of_particles=args.people,
                                      random_seed=args.seed))
