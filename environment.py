import numpy as np
import pymunk
import robot
import math

class WalkerEnv:
    def __init__(self, width = 800, height = 600, time_limit=20, substeps=7, max_torque=100000.0, seed=0, number_of_ground_points=20, fps=30):
        self.W = width
        self.H = height
        self.time_limit = time_limit
        self.substeps = substeps
        self.max_torque = max_torque
        self.rng = np.random.default_rng(seed)
        self.number_of_ground_points = number_of_ground_points
        self.fps = fps
        self.space = None
        self.points = None
        self.door = None
        self.robot = None


    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -800)
        self.space.iterations = 100

        self._create_environment()

        door_x = self.rng.integers(int(self.W*0.8), int(self.W*0.9))
        door_h = 250
        door_w = 100
        self.door = (door_x, door_w, door_h)
        self.robot = self._create_robot()

        self.frame_count = 0
        self.time_count = 0

    def step(self, action):
        action = np.clip(action, -1, 1)
        torque = action * self.max_torque
        self.robot.command_torque = torque

        dt = 1/self.fps
        for _ in range(self.substeps):
            self.robot.apply_joint_torque()
            self.space.step(dt / self.substeps)

        self.frame_count += 1
        self.time_count += dt

    def _create_environment(self):

        points_max_height = int(self.H * 0.12)
        points = []
        for i in range(self.number_of_ground_points):
            x = self.W // (self.number_of_ground_points +1) * i + self.rng.integers(-self.W//(2*self.number_of_ground_points+5), self.W//(2*self.number_of_ground_points+5))
            y = self.rng.integers(10, points_max_height)
            points.append((x, y))
        points = sorted(points, key=lambda point: point[0])
        points[0] = (0, points[0][1])
        points[-1] = (self.W, points[-1][1]) 
        static_lines = []
        for i in range(len(points) - 1):
            segment_shape = pymunk.Segment(self.space.static_body, points[i], points[i + 1], 1.0)
            segment_shape.friction = 0.5
            segment_shape.elasticity = 0.3
            static_lines.append(segment_shape)
            self.space.add(segment_shape)
        
        for i in range(1, len(static_lines) - 1):
            static_lines[i].set_neighbors(points[i - 1], points[i + 2])

        boundary = []
        left = pymunk.Segment(self.space.static_body, (0, points[0][1]), (0, self.H), 1.0)
        right = pymunk.Segment(self.space.static_body, (self.W, points[-1][1]), (self.W, self.H), 1.0)
        top = pymunk.Segment(self.space.static_body, (0, self.H), (self.W, self.H), 1.0)
        boundary.append(left)
        boundary.append(right)
        boundary.append(top)
        self.space.add(*boundary)
        self.points = points
        self.static_lines = static_lines
        return None

    def _create_robot(self):
        intial_pos = (self.rng.integers(200,400), self.rng.integers(200,300))
        intial_angle = self.rng.uniform(0, 2*math.pi)
        self.robot = robot.TwoLinkRobot(self.space, arm1_length=100, arm2_length=100, arm1_mass=1.0, arm2_mass=1.0,
         joint_pos=intial_pos, arm1_angle=intial_angle, torque_limit = self.max_torque)
        return self.robot


    def get_replay_step_data(self, action):
        """Get minimal data to save per step for replay."""
        return {
            "a": float(action),
            "vertices1_world": self.robot.get_state()[0],
            "vertices2_world": self.robot.get_state()[1],
            "joint_pos": self.robot.get_state()[2],
        }
