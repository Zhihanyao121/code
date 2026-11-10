import numpy as np
from particle import Particle
from forces import compute_density, compute_pressure, compute_accelerations


class CrowdSimulation:
    """SPH crowd with circular bodies, wall corners, and local avoidance."""

    def __init__(self, width=12.0, height=3.6, number_of_particles=100,
                 exit_width=1.2, random_seed=42, radius=0.22):
        if width <= 2 * radius or height <= 2 * radius:
            raise ValueError('Room too small for the pedestrian radius')
        self.width = float(width)
        self.height = float(height)
        self.radius = float(radius)
        self.number_of_particles = int(number_of_particles)
        self.exit_width = exit_width
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.h = 1.25
        self.free_speed = 1.3
        self.critical_density = 1.35
        self.jam_density = 3.7
        self.tau = 0.5
        self.pressure_coefficient = 4.0
        self.max_speed = 2.0
        self.elapsed_time = 0.0
        self.evacuation_time = None
        self.statistics_printed = False
        self.particles = []
        self.finished_particles = []
        self.exits = []
        self.stuck_events = 0
        self.recovered_events = 0
        self.stuck_threshold = 2.0
        self.recovery_duration = 1.5
        # Adaptive queue: managed independently for each opening.
        self.queue_enabled = True
        self.queue_trigger_seconds = 1.5
        self.queue_release_seconds = 3.0
        self.queue_min_active_seconds = 3.0
        self.queue_activations = 0
        self.queue_states = []
        self._configure_exits()
        self.create_random_particles(self.number_of_particles)
        self._initialize_queue_states()

    def _initialize_queue_states(self):
        self.queue_states = [dict(active=False, congested_for=0.0,
                                  clear_for=0.0, active_for=0.0,
                                  nearby=0, stuck=0, mean_speed=0.0)
                             for _ in self.exits]

    def _configure_exits(self):
        if self.exit_width is None:
            # Normal demo mode: three openings on the right wall.
            if self.height < 3.6:
                raise ValueError('Three-exit mode needs height >= 3.6')
            openings = [(0.1, 0.9), (1.35, 2.25), (2.7, 3.5)]
        else:
            if not 2 * self.radius < self.exit_width < self.height:
                raise ValueError('Exit width must exceed body diameter and be below room height')
            center = self.height / 2.0
            openings = [(center - self.exit_width / 2.0,
                         center + self.exit_width / 2.0)]
        self.exits = [dict(side='right', start=a, end=b, count=0)
                      for a, b in openings]

    def create_random_particles(self, count):
        # A compact starting group, identical for every tested exit width.
        spacing = 2 * self.radius + 0.055
        lanes = max(1, int((self.height - 2 * self.radius - 0.15) // spacing) + 1)
        columns = int(np.ceil(count / lanes))
        x_start = 0.65
        if x_start + (columns - 1) * spacing + self.radius >= self.width - 0.5:
            raise ValueError('Not enough room for initial crowd')
        y_start = (self.height - (lanes - 1) * spacing) / 2.0
        positions = [(x_start + c * spacing, y_start + row * spacing)
                     for c in range(columns) for row in range(lanes)][:count]
        self.rng.shuffle(positions)
        self.particles = []
        for index, (x, y) in enumerate(positions):
            jitter = self.rng.uniform(-0.012, 0.012, size=2)
            p = Particle([x + jitter[0], y + jitter[1]], mass=1.0)
            p.pedestrian_id = index
            p.progress_x = float(p.position[0])
            p.last_progress_time = 0.0
            p.stuck = False
            p.stuck_since = None
            p.recovery_until = 0.0
            p.yield_until = 0.0
            self.particles.append(p)

    def reset(self, number_of_particles=None):
        if number_of_particles is not None:
            self.number_of_particles = int(number_of_particles)
        self.rng = np.random.default_rng(self.random_seed)
        self.elapsed_time = 0.0
        self.evacuation_time = None
        self.statistics_printed = False
        self.stuck_events = 0
        self.recovered_events = 0
        self.finished_particles = []
        self._configure_exits()
        self.create_random_particles(self.number_of_particles)
        self.queue_activations = 0
        self._initialize_queue_states()

    def get_exit_center(self, exit_info):
        return np.array([self.width + 1.0,
                         (exit_info['start'] + exit_info['end']) / 2.0])

    def choose_exit(self, particle):
        distances = [np.linalg.norm(self.get_exit_center(e) - particle.position)
                     for e in self.exits]
        index = int(np.argmin(distances))
        return index, self.get_exit_center(self.exits[index])

    def _valid_exit(self, y):
        for i, e in enumerate(self.exits):
            if e['start'] + self.radius <= y <= e['end'] - self.radius:
                return i
        return None

    def _right_wall_segments(self):
        beginning = 0.0
        for e in sorted(self.exits, key=lambda item: item['start']):
            if e['start'] > beginning:
                yield beginning, e['start']
            beginning = max(beginning, e['end'])
        if beginning < self.height:
            yield beginning, self.height

    def _wall_constraints(self, p):
        """Circle-vs-segment projection includes both corners of each exit."""
        r = self.radius
        pos = p.position
        vel = p.velocity
        # Three continuous walls; only right wall contains exits.
        if pos[0] < r:
            pos[0] = r
            vel[0] = max(0.0, vel[0])
        if pos[1] < r:
            pos[1] = r
            vel[1] = max(0.0, vel[1])
        if pos[1] > self.height - r:
            pos[1] = self.height - r
            vel[1] = min(0.0, vel[1])

        for start, end in self._right_wall_segments():
            nearest = np.array([self.width, np.clip(pos[1], start, end)])
            delta = pos - nearest
            distance = np.linalg.norm(delta)
            if distance < r:
                if distance < 1e-12:
                    normal = np.array([-1.0, 0.0])
                else:
                    normal = delta / distance
                pos[:] = nearest + normal * r
                approaching = np.dot(vel, normal)
                if approaching < 0:
                    vel[:] -= approaching * normal  # sliding, not bouncing

    def _local_avoidance(self, p):
        """Steer around pedestrians ahead, with more weight near the exit."""
        x, y = p.position
        nearest_exit = min(self.exits,
                           key=lambda e: abs(y - (e['start'] + e['end']) / 2))
        center_y = (nearest_exit['start'] + nearest_exit['end']) / 2.0
        distance_to_wall = self.width - x
        steering = np.zeros(2)

        # Approach the opening *before* reaching the wall/corners.
        if distance_to_wall < 3.0:
            proximity = np.clip((3.0 - distance_to_wall) / 3.0, 0.0, 1.0)
            alignment = np.clip((center_y - y) * 3.0, -2.5, 2.5)
            steering[1] += proximity * alignment

        # Look at people ahead within a small neighborhood.
        # Side selection favors movement toward the middle of the opening.
        for other in self.particles:
            if other is p:
                continue
            dx = other.position[0] - x
            dy = other.position[1] - y
            if not (0 < dx < 0.95 and abs(dy) < 0.60):
                continue
            gap = max(np.hypot(dx, dy), 0.01)
            weight = (0.95 - dx) / 0.95 * (0.60 - abs(dy)) / 0.60
            if abs(dy) > 0.035:
                side = -np.sign(dy)  # move away from the blocker
            else:
                side = np.sign(center_y - y)
                if side == 0:
                    side = 1.0 if (p.pedestrian_id % 2) else -1.0
            # Avoid steering directly into the side walls.
            if (side < 0 and y <= 2.5 * self.radius) or (
                    side > 0 and y >= self.height - 2.5 * self.radius):
                side = -side
            steering[1] += 1.2 * weight * side / max(gap, 0.25)

        return np.clip(steering, -3.0, 3.0)

    # =================================================
    # STAGNATION DETECTION AND DETERMINISTIC YIELDING
    # =================================================

    def _update_stuck_states(self):
        """Only consider walkers close to an exit; distinguish queuing
        from long-lasting lack of forward progress."""
        for p in self.particles:
            if p.position[0] >= p.progress_x + 0.20:
                p.progress_x = float(p.position[0])
                p.last_progress_time = self.elapsed_time
                if p.stuck:
                    self.recovered_events += 1
                p.stuck = False
                p.stuck_since = None
            elif p.position[0] < p.progress_x - 0.4:
                # Reset tracking if moved substantially backward.
                p.progress_x = float(p.position[0])
                p.last_progress_time = self.elapsed_time

            near_exit = p.position[0] >= self.width - 2.5
            stalled = (self.elapsed_time - p.last_progress_time
                       >= self.stuck_threshold)
            if near_exit and stalled and not p.stuck:
                p.stuck = True
                p.stuck_since = self.elapsed_time
                self.stuck_events += 1
            if not near_exit:
                p.stuck = False
                p.stuck_since = None

    def _recovery_steering(self, p):
        """Side-step or yield; never teleport through neighbors or walls."""
        if not p.stuck:
            return np.zeros(2)

        center_y = (self.exits[p.exit_index]['start']
                    + self.exits[p.exit_index]['end']) / 2.0
        ahead = []
        for other in self.particles:
            if other is p:
                continue
            dx = other.position[0] - p.position[0]
            dy = other.position[1] - p.position[1]
            if -0.12 < dx < 0.85 and abs(dy) < 0.55:
                ahead.append(other)

        if not ahead:
            # Free path: deliberately re-accelerate and align with door.
            return np.array([1.0, np.clip(2.0 * (center_y - p.position[1]), -2, 2)])

        # A pedestrian closer to the door (or lower stable ID on tie)
        # has priority. Others slow down and move sideways to make room.
        contenders = ahead + [p]
        leader = min(contenders,
                     key=lambda q: (-round(float(q.position[0]), 2),
                                    q.pedestrian_id))
        if leader is p:
            return np.array([1.5, np.clip(2.5 * (center_y - p.position[1]), -2, 2)])

        # Stable direction, prefer opening center but avoid wall edge.
        side = np.sign(center_y - p.position[1])
        if side == 0:
            side = 1 if p.pedestrian_id % 2 else -1
        if side > 0 and p.position[1] > self.height - 3*self.radius:
            side = -1
        if side < 0 and p.position[1] < 3*self.radius:
            side = 1
        # Negative longitudinal acceleration creates a space for the leader.
        return np.array([-1.8, 2.0 * side])

    # =================================================
    # ADAPTIVE QUEUE: DETECT / ACTIVATE / RELEASE
    # =================================================

    def _update_queue_states(self, dt):
        """Detect sustained congestion for every exit independently.

        Requires at least three stalled walkers near the exit, plus either
        low average speed or a higher stalled count. Uses hysteresis so
        the control does not oscillate at every time step.
        """
        for exit_index, state in enumerate(self.queue_states):
            nearby = [p for p in self.particles
                      if p.exit_index == exit_index
                      and self.width - 2.0 <= p.position[0] <= self.width + self.radius]
            stuck = sum(bool(p.stuck) for p in nearby)
            mean_speed = (float(np.mean([np.linalg.norm(p.velocity) for p in nearby]))
                          if nearby else 0.0)
            state['nearby'] = len(nearby)
            state['stuck'] = stuck
            state['mean_speed'] = mean_speed

            congested = (len(nearby) >= 3 and stuck >= 3
                         and (mean_speed < 0.35 or stuck >= 5))
            if congested:
                state['congested_for'] += dt
                state['clear_for'] = 0.0
            else:
                state['clear_for'] += dt
                state['congested_for'] = 0.0

            if state['active']:
                state['active_for'] += dt
                if (state['active_for'] >= self.queue_min_active_seconds
                        and state['clear_for'] >= self.queue_release_seconds):
                    state['active'] = False
                    state['active_for'] = 0.0
            elif (self.queue_enabled
                  and state['congested_for'] >= self.queue_trigger_seconds):
                state['active'] = True
                state['active_for'] = 0.0
                state['clear_for'] = 0.0
                self.queue_activations += 1
                print(f'[{self.elapsed_time:.1f}s] Exit {exit_index + 1}: QUEUE ON')

    def _queue_steerings(self):
        """Ordered lane-guidance accelerations for active exits.

        Queue steering is a behavioral control, never a position teleport.
        Circle and wall contacts remain active afterwards.
        """
        commands = {p.pedestrian_id: np.zeros(2) for p in self.particles}
        for exit_index, state in enumerate(self.queue_states):
            if not state['active']:
                continue
            e = self.exits[exit_index]
            width = e['end'] - e['start']
            diameter = 2 * self.radius
            lane_spacing = diameter + 0.08
            usable_width = width - diameter
            lane_count = max(1, 1 + int(max(0.0, usable_width) // lane_spacing))
            center_y = (e['start'] + e['end']) / 2.0
            lane_y = [center_y + (i - (lane_count - 1) / 2) * lane_spacing
                      for i in range(lane_count)]
            # Guide pedestrians in the 3m approach zone only.
            nearby = [p for p in self.particles
                      if p.exit_index == exit_index
                      and self.width - 3.0 < p.position[0] < self.width + self.radius]
            if not nearby:
                continue
            # Preserve approximate lateral lanes, select closest lane.
            groups = [[] for _ in lane_y]
            for p in nearby:
                k = int(np.argmin([abs(p.position[1] - ly) for ly in lane_y]))
                groups[k].append(p)
            for lane_index, people in enumerate(groups):
                # Frontmost pedestrian first; stable ID breaks ties.
                people.sort(key=lambda p: (-p.position[0], p.pedestrian_id))
                leader = None
                for p in people:
                    distance_to_door = max(0.0, self.width - p.position[0])
                    approach = np.clip((3.0 - distance_to_door) / 3.0, 0.0, 1.0)
                    lateral = np.clip(3.0 * (lane_y[lane_index] - p.position[1]), -2.0, 2.0)
                    forward = 0.4
                    if leader is not None:
                        gap = float(leader.position[0] - p.position[0])
                        safe_gap = diameter + 0.12
                        if gap < safe_gap + 0.30:
                            forward = -3.5 * np.clip((safe_gap + 0.30 - gap) / 0.30, 0, 1)
                    # The leading pedestrian should be encouraged toward the opening.
                    commands[p.pedestrian_id] += np.array([
                        forward * approach,
                        lateral * approach
                    ])
                    leader = p
        return commands

    def queue_diagnostics(self):
        return {
            'queue_active': any(s['active'] for s in self.queue_states),
            'active_exits': [i + 1 for i, s in enumerate(self.queue_states) if s['active']],
            'queue_activations': self.queue_activations,
            'per_exit': [dict(state) for state in self.queue_states]
        }

    def collision_diagnostics(self):
        """Reports maximum body overlap (meters) and stalled count."""
        overlap = 0.0
        for i, a in enumerate(self.particles):
            for b in self.particles[i+1:]:
                overlap = max(overlap, 2*self.radius - np.linalg.norm(
                    a.position-b.position))
        return {
            'max_overlap_m': max(0.0, float(overlap)),
            'stuck_now': sum(p.stuck for p in self.particles),
            'stuck_events': self.stuck_events,
            'recovered_events': self.recovered_events,
            'queue_active': any(s['active'] for s in self.queue_states),
            'queue_activations': self.queue_activations,
        }

    def _resolve_collisions(self):
        diameter = 2 * self.radius
        people = self.particles
        for _ in range(6):
            for i in range(len(people)):
                for j in range(i + 1, len(people)):
                    a, b = people[i], people[j]
                    delta = b.position - a.position
                    d = np.linalg.norm(delta)
                    if d >= diameter:
                        continue
                    if d < 1e-12:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = delta / d
                    shift = 0.5 * (diameter - d) * direction
                    a.position -= shift
                    b.position += shift
            for p in people:
                self._wall_constraints(p)

    def step(self, dt):
        if dt <= 0:
            raise ValueError('dt must be positive')
        if not self.particles:
            return
        compute_density(self.particles, self.h)
        compute_pressure(self.particles, self.critical_density,
                         self.pressure_coefficient)
        targets = []
        for p in self.particles:
            p.exit_index, target = self.choose_exit(p)
            targets.append(target)
        compute_accelerations(self.particles, self.h, targets,
                              self.free_speed, self.critical_density,
                              self.jam_density, self.tau)
        old_positions = [p.position.copy() for p in self.particles]
        self._update_stuck_states()
        self._update_queue_states(dt)
        queue_commands = self._queue_steerings()
        steering = []
        for p in self.particles:
            if self.queue_states[p.exit_index]['active'] and p.pedestrian_id in queue_commands:
                # During queue control do not also execute lateral avoidance,
                # which could compete with alignment and induce oscillations.
                extra = queue_commands[p.pedestrian_id]
            else:
                extra = self._local_avoidance(p) + self._recovery_steering(p)
            steering.append(extra)
        for p, extra in zip(self.particles, steering):
            p.acceleration += extra
            p.velocity += p.acceleration * dt
            speed = np.linalg.norm(p.velocity)
            if speed > self.max_speed:
                p.velocity *= self.max_speed / speed
            p.position += p.velocity * dt
            self._wall_constraints(p)
        self._resolve_collisions()
        for p, old in zip(self.particles, old_positions):
            # Recompute velocity to reflect contact; bound numerical spikes.
            p.velocity = (p.position - old) / dt
            speed = np.linalg.norm(p.velocity)
            if speed > self.max_speed:
                p.velocity *= self.max_speed / speed
        for p in self.particles[:]:
            if p.position[0] > self.width + self.radius:
                exit_index = self._valid_exit(p.position[1])
                if exit_index is not None:
                    self.particles.remove(p)
                    self.finished_particles.append(p)
                    self.exits[exit_index]['count'] += 1
        self.elapsed_time += dt
        if not self.particles and self.evacuation_time is None:
            self.evacuation_time = self.elapsed_time
            if not self.statistics_printed:
                self.print_statistics()
                self.statistics_printed = True

    def print_statistics(self):
        print(f'Finished: {len(self.finished_particles)}/{self.number_of_particles}; '
              f'Time: {self.elapsed_time:.2f}s; '
              f'Exit counts: {[e["count"] for e in self.exits]}; '
              f'Stuck events: {self.stuck_events}; '
              f'Recoveries: {self.recovered_events}; '
              f'Queue activations: {self.queue_activations}')
