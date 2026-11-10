import numpy as np


class Particle:
    def __init__(self, position, velocity=None, mass=1.0):
        self.position = np.array(position, dtype=float)

        if velocity is None:
            velocity = [0.0, 0.0]

        self.velocity = np.array(velocity, dtype=float)

        self.mass = mass

        self.density = 0.0
        self.pressure = 0.0
        self.desired_speed = 0.0
        self.acceleration = np.zeros(2)

        self.exit_index = None