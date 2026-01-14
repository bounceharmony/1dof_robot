import pymunk
import math

class TwoLinkRobot:
    """
    Two rigid links connected by a revolute joint at `joint_pos`.

    Coordinate convention:
      - Angles are in radians, standard math: 0 along +x, CCW positive.
      - joint_pos is the elbow joint location between the two links.
    """

    def __init__(
        self,
        space: pymunk.Space,
        arm1_length: float,
        arm2_length: float,
        arm1_mass: float,
        arm2_mass: float,
        joint_pos: tuple[float, float],
        arm1_angle: float,
        arm2_angle: float | None = None,
        thickness: float = 10.0,
        friction: float = 0.7,
        elasticity: float = 0.3,
        limit_angles: tuple[float, float] | None = None,
        torque_limit: float | None = None,
    ):
        self.space = space
        self.arm1_length = arm1_length
        self.arm2_length = arm2_length
        self.thickness = thickness
        self.torque_limit = torque_limit
        self.command_torque = 0
        jx, jy = joint_pos

        if arm2_angle is None:
            arm2_angle = arm1_angle + math.pi / 2.0

        moment1 = pymunk.moment_for_box(arm1_mass, (arm1_length, thickness))
        moment2 = pymunk.moment_for_box(arm2_mass, (arm2_length, thickness))

        self.body1 = pymunk.Body(arm1_mass, moment1)
        self.body2 = pymunk.Body(arm2_mass, moment2)

        c1x = jx + (arm1_length / 2.0) * math.cos(arm1_angle)
        c1y = jy + (arm1_length / 2.0) * math.sin(arm1_angle)
        c2x = jx + (arm2_length / 2.0) * math.cos(arm2_angle)
        c2y = jy + (arm2_length / 2.0) * math.sin(arm2_angle)

        self.body1.position = (c1x, c1y)
        self.body2.position = (c2x, c2y)
        self.body1.angle = arm1_angle
        self.body2.angle = arm2_angle

        half_length1 = arm1_length / 2.0
        half_thickness = thickness / 2.0
        half_length2 = arm2_length / 2.0
        
        vertices1 = [
            (-half_length1, -half_thickness),
            (half_length1, -half_thickness),
            (half_length1, half_thickness),
            (-half_length1, half_thickness)
        ]
        
        vertices2 = [
            (-half_length2, -half_thickness),
            (half_length2, -half_thickness),
            (half_length2, half_thickness),
            (-half_length2, half_thickness)
        ]
        
        self.shape1 = pymunk.Poly(self.body1, vertices1)
        self.shape2 = pymunk.Poly(self.body2, vertices2)

        for s in (self.shape1, self.shape2):
            s.friction = friction
            s.elasticity = elasticity

        joint_world = pymunk.Vec2d(jx, jy)

        self.elbow = pymunk.PivotJoint(self.body1, self.body2, joint_world)
        self.elbow.collide_bodies = False

        self.elbow.max_force = 1e10
        self.elbow.error_bias = 1e-3
        self.elbow.max_bias = 1e6 

        self.rot_spring = pymunk.DampedRotarySpring(self.body1, self.body2, rest_angle=math.pi/3, stiffness=0, damping=1e3)

        self.limit = None
        if limit_angles is not None:
            min_a, max_a = limit_angles
            self.limit = pymunk.RotaryLimitJoint(self.body1, self.body2, min_a, max_a)

        add_list = [self.body1, self.body2, self.shape1, self.shape2, self.elbow, self.rot_spring]
        if self.limit is not None:
            add_list.append(self.limit)
        space.add(*add_list)

    def apply_joint_torque(self) -> None:
        """Apply internal torque at the elbow joint."""
        if self.torque_limit is not None: 
            self.command_torque = max(-self.torque_limit, min(self.torque_limit, self.command_torque))
        self.body1.torque += self.command_torque
        self.body2.torque -= self.command_torque

    def get_joint_world_pos(self) -> tuple[float, float]:
        """Get world position of the elbow joint point."""
        return self.body1.local_to_world((-self.arm1_length / 2.0, 0))
    
    def get_state(self):
        vertices1_world = [self.body1.local_to_world(v) for v in self.shape1.get_vertices()]
        vertices2_world = [self.body2.local_to_world(v) for v in self.shape2.get_vertices()]
        
        joint_pos = self.get_joint_world_pos()
        return vertices1_world, vertices2_world, joint_pos