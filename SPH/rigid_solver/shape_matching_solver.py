import taichi as ti
import numpy as np
from ..containers import BaseContainer

# this script is not used in the project
@ti.data_oriented
class ShapeMatchingRigidSolver():
    def __init__(self, dt, container: BaseContainer):
        self.container = container
        self.dt = dt
        self.rigid_particle_temp_positions = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_centers_of_mass = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_rotation_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=10)


    @ti.kernel
    def update_rigid_velocities(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_rigid_positions(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                self.rigid_particle_temp_positions[p_i] = self.container.particle_positions[p_i] + self.dt[None] * self.container.particle_velocities[p_i]
    
    # @ti.func
    # def compute_com(self, object_id):
    #     cm = ti.Vector([0.0, 0.0, 0.0])
    #     for p_i in range(self.container.particle_num[None]):
    #         if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_ids[p_i] == object_id:
    #             mass = self.container.V0 * self.container.particle_densities[p_i]
    #             cm += mass * self.container.particle_positions[p_i]
    #     cm /= self.container.rigid_body_masses[object_id]
    #     return cm
    
    # @ti.kernel
    # def solve_constraints(self, object_id: int):
    #     # compute center of mass
    #     cm = self.compute_com(object_id)
    #     # A
    #     A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #     for p_i in range(self.container.particle_num[None]):
    #         if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_ids[p_i] == object_id:
    #             q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]
    #             p = self.container.particle_positions[p_i] - cm
    #             A += self.container.V0 * self.container.particle_densities[p_i] * p.outer_product(q)

    #     R, S = ti.polar_decompose(A)
        
    #     if all(abs(R) < 1e-6):
    #         R = ti.Matrix.identity(ti.f32, 3)
        
    #     for p_i in range(self.container.particle_num[None]):
    #         if self.container.is_dynamic_rigid_body(p_i) and self.container.particle_object_ids[p_i] == object_id:
    #             goal = cm + R @ (self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id])
    #             self.container.particle_positions[p_i] = goal
        

    @ti.kernel
    def compute_temp_center_of_mass(self):
        self.rigid_body_temp_centers_of_mass.fill(0.0)

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                self.rigid_body_temp_centers_of_mass[object_id] += self.container.V0 * self.container.particle_densities[p_i] * self.rigid_particle_temp_positions[p_i]

        for obj_i in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                self.rigid_body_temp_centers_of_mass[obj_i] /= self.container.rigid_body_masses[obj_i]
               
    @ti.kernel
    def solve_constraints(self):
        self.rigid_body_temp_rotation_matrices.fill(0.0)

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]
                p = self.rigid_particle_temp_positions[p_i] - self.rigid_body_temp_centers_of_mass[object_id]
                self.rigid_body_temp_rotation_matrices[object_id] += self.container.V0 * self.container.particle_densities[p_i] * p.outer_product(q)

        for obj_i in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                A_pq = self.rigid_body_temp_rotation_matrices[obj_i]
                R, S = ti.polar_decompose(A_pq)
                if all(abs(R) < 1e-6):
                    R = ti.Matrix.identity(ti.f32, 3)
                self.rigid_body_temp_rotation_matrices[obj_i] = R

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                goal = self.rigid_body_temp_centers_of_mass[object_id] + self.rigid_body_temp_rotation_matrices[object_id] @ (self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id])
                # self.container.particle_velocities[p_i] = (goal - self.container.particle_positions[p_i]) / self.dt[None]
                self.container.particle_positions[p_i] = goal