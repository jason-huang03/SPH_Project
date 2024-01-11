import pybullet as p
import pybullet_data
import taichi as ti
import numpy as np
import os
import math
from ..containers import BaseContainer
from ..utils import create_urdf
from typing import List, Tuple, Dict, Union

# Note that the index returned by bullet is not the same as the index in the container. So we need to maintain a mapping between the two indices.
# ! Note I did not implement changing time step in the middle of the simulation. So the time step is fixed to be the same as the one in the container.
# ! also, we currently assume that the center of mass is exactly the base position of the object. This is not true in general.
class PyBulletSolver():
    def __init__(self, container: BaseContainer, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        self.container = container
        self.total_time = 0.0
        self.present_rigid_object = []
        assert container.dim == 3, "PyBulletSolver only supports 3D simulation currently"

        self.cfg = container.cfg
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        num_rigid_bodies = len(self.rigid_bodies) + len(self.rigid_blocks)
        self.dt = dt

        # mapping between container index and bullet index
        self.container_idx_to_bullet_idx = {}
        self.bullet_idx_to_container_idx = {}

        if num_rigid_bodies != 0:
            self.physicsClient = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setTimeStep(self.dt)
            p.setGravity(*gravity)
            ground_rotation = p.getQuaternionFromEuler([- math.pi / 2, 0, 0]) # make y axis as up axis
            # planeId = p.loadURDF("plane.urdf", baseOrientation=ground_rotation)
            self.create_boundary()

        else:
            self.physicsClient = None
            print("No rigid body in the scene, skip bullet solver initialization.")

        # we insert rigid body in the solver each round. so we do not call insert_rigid_object here.

    def insert_rigid_object(self):
        for rigid_body in self.rigid_bodies:
            self.init_rigid_body(rigid_body)

        for rigid_block in self.rigid_blocks:
            self.init_rigid_block(rigid_block)

    def create_boundary(self, thickness: float = 0.01):
        # this function create the domain boundary

        # we do not want the rigid to hit the boundary of fluid so we move each wall inside a little bit.
        eps = self.container.padding + self.container.particle_diameter + self.container.domain_box_thickness 
        domain_start = self.container.domain_start
        domain_end = self.container.domain_end
        domain_start = np.array(domain_start) + eps
        domain_end = np.array(domain_end) - eps
        domain_size = self.container.domain_size
        domain_center = (domain_start + domain_end) / 2

        # Creating each wall
        self.create_wall(position=[domain_center[0], domain_start[1] - thickness / 1.9, domain_center[2]], half_extents=[domain_size[0] / 2, thickness / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_center[0], domain_end[1] + thickness / 1.9, domain_center[2]], half_extents=[domain_size[0] / 2, thickness / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_start[0] - thickness / 1.9, domain_center[1], domain_center[2]], half_extents=[thickness / 2, domain_size[1] / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_end[0] + thickness / 1.9, domain_center[1], domain_center[2]], half_extents=[thickness / 2, domain_size[1] / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_center[0], domain_center[1], domain_start[2] - thickness / 1.9], half_extents=[domain_size[0] / 2, domain_size[1] / 2, thickness / 2])
        self.create_wall(position=[domain_center[0], domain_center[1], domain_end[2] + thickness / 1.9], half_extents=[domain_size[0] / 2, domain_size[1] / 2, thickness / 2])

    def create_wall(self, position, half_extents):
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=position)

    def init_rigid_body(self, rigid_body):
        container_idx = rigid_body["objectId"]

        # dealing with entry time
        if container_idx in self.present_rigid_object:
            return
        if rigid_body["entryTime"] > self.total_time:
            return

        is_dynamic = rigid_body["isDynamic"]
        if is_dynamic:
            velocity = np.array(rigid_body["velocity"], dtype=np.float32)
        else:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        mesh_file_path = rigid_body["geometryFile"]
        urdf_file_path = mesh_file_path[:-4] + ".urdf"
        scale = rigid_body["scale"]
        mass = self.container.rigid_body_masses[container_idx]
        
        # create a temporary urdf file to load the mesh
        create_urdf(mesh_file_path, mass, scale, urdf_file_path)

        translation = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * (2 * math.pi)
        direction = rigid_body["rotationAxis"]
        rotation_euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
        rotation_quaternion = p.getQuaternionFromEuler(rotation_euler)
        rotation_matrix = p.getMatrixFromQuaternion(rotation_quaternion)
        rotation_matrix = np.array(rotation_matrix).reshape((3, 3))

        bullet_idx = p.loadURDF(urdf_file_path, basePosition=translation, baseOrientation=rotation_quaternion)

        # delete the urdf file after loading it
        os.remove(urdf_file_path)

        self.container_idx_to_bullet_idx[container_idx] = bullet_idx
        self.bullet_idx_to_container_idx[bullet_idx] = container_idx

        if is_dynamic:
            self.container.rigid_body_original_centers_of_mass[container_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_centers_of_mass[container_idx] = translation
            self.container.rigid_body_rotations[container_idx] = rotation_matrix
            self.container.rigid_body_velocities[container_idx] = velocity
            self.container.rigid_body_angular_velocities[container_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p.resetBaseVelocity(bullet_idx, velocity)

        if not is_dynamic:
            p.changeDynamics(bullet_idx, -1, mass=0.0)

        self.present_rigid_object.append(container_idx)
        
    def init_rigid_block(self, rigid_block):
        # TODO enable adding rigid block
        raise NotImplementedError

    def apply_force(self, container_idx, force: Tuple[float, float, float]):
        # ! here we assume the center of mass is exactly the base position
        bullet_idx = self.container_idx_to_bullet_idx[container_idx]
        com_pos, _ = p.getBasePositionAndOrientation(bullet_idx)
        p.applyExternalForce(bullet_idx, -1, forceObj=force, posObj=com_pos, flags=p.WORLD_FRAME)        

    def apply_torque(self, container_idx, torque: Tuple[float, float, float]):
        bullet_idx = self.container_idx_to_bullet_idx[container_idx]
        p.applyExternalTorque(bullet_idx, -1, torqueObj=torque, flags=p.WORLD_FRAME)

    def step(self):
        if self.physicsClient is None:
            return

        # apply force and torque to each rigid body
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                force_i = self.container.rigid_body_forces[container_id]
                torque_i = self.container.rigid_body_torques[container_id]
                self.apply_force(container_id, force_i)
                self.apply_torque(container_id, torque_i)
                self.container.rigid_body_forces[container_id] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[container_id] = np.array([0.0, 0.0, 0.0])
        
        p.stepSimulation()

        # update rigid body states in the container. updating rigid particle state from center of mass and rotation is done in fluid solver.
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                state_i = self.get_rigid_body_states(container_id)
                self.container.rigid_body_centers_of_mass[container_id] = state_i["position"]
                self.container.rigid_body_rotations[container_id] = state_i["rotation_matrix"]
                self.container.rigid_body_velocities[container_id] = state_i["linear_velocity"]
                self.container.rigid_body_angular_velocities[container_id] = state_i["angular_velocity"]
        
    def get_rigid_body_states(self, container_idx):
        # ! here we use the information of base frame. We assume the center of mass is exactly the base position.
        bullet_idx = self.container_idx_to_bullet_idx[container_idx]
        linear_velocity, angular_velocity = p.getBaseVelocity(bullet_idx)
        position, orientation = p.getBasePositionAndOrientation(bullet_idx)

        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))
            
        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "position": position,
            "rotation_matrix": rotation_matrix
        }