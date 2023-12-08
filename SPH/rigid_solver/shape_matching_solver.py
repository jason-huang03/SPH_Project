import taichi as ti
import numpy as np
from ..containers import BaseContainer

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
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_is_dynamic[p_i]:
                    self.container.particle_velocities[p_i] += self.dt[None] * ti.Vector([0, -9.8, 0])

                    self.rigid_particle_temp_positions[p_i] = self.container.particle_positions[p_i] + self.dt[None] * self.container.particle_velocities[p_i]

    

        # self.rigid_body_temp_centers_of_mass.fill(0)
        # for p_i in range(self.container.particle_num[None]):
        #     if self.container.particle_materials[p_i] == self.container.material_rigid:
        #         obj_id = self.container.particle_object_ids[p_i]
        #         self.rigid_body_temp_centers_of_mass[obj_id] += (
        #             self.rigid_particle_temp_positions[p_i] 
        #             * self.container.V0 
        #             * self.container.particle_densities[p_i]
        #         )

        # for obj_id in range(self.container.rigid_body_num[None]):
        #     self.rigid_body_temp_centers_of_mass[obj_id] /= self.container.rigid_body_masses[obj_id]

        
        # for p_i in range(self.container.particle_num[None]):
        #     if self.container.particle_materials[p_i] == self.container.material_rigid:
        #         obj_id = self.container.particle_object_ids[p_i]
        #         self.rigid_particle_temp_positions[p_i] = self.rigid_body_temp_centers_of_mass[obj_id] + self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[obj_id]
        #         self.container.particle_velocities[p_i] = (self.rigid_particle_temp_positions[p_i] - self.container.particle_positions[p_i]) / self.dt[None]


        # self.rigid_body_temp_rotation_matrices.fill(0)
        # for p_i in range(self.container.particle_num[None]):
        #     if self.container.particle_materials[p_i] == self.container.material_rigid:
        #         obj_id = self.container.particle_object_ids[p_i]

        #         p = self.rigid_particle_temp_positions[p_i] - self.rigid_body_temp_centers_of_mass[obj_id]
        #         q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[obj_id]
        #         self.rigid_body_temp_rotation_matrices[obj_id] += p.outer_product(q) * self.container.V0 * self.container.particle_densities[p_i]

        # for obj_id in range(self.container.rigid_body_num[None]):
        #     A_pq = self.rigid_body_temp_rotation_matrices[obj_id]
        #     R, S = ti.polar_decompose(A_pq)
        #     if all(abs(R) < 1e-6):
        #         R = ti.Matrix.identity(ti.f32, 3)
        #     self.rigid_body_temp_rotation_matrices[obj_id] = R

        # for p_i in range(self.container.particle_num[None]):
        #     if self.container.particle_materials[p_i] == self.container.material_rigid:
        #         obj_id = self.container.particle_object_ids[p_i]
        #         goal_pos = (
        #             self.rigid_body_temp_rotation_matrices[obj_id]
        #             @ (self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[obj_id])
        #             + self.rigid_body_temp_centers_of_mass[obj_id]
        #         )

        #         self.container.particle_velocities[p_i] = (goal_pos - self.container.particle_positions[p_i]) / self.dt[None]
    


    @ti.kernel
    def update_rigid_positions(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_is_dynamic[p_i]:
                    self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]

        for obj_id in range(self.container.rigid_body_num[None]):
            if self.container.rigid_body_is_dynamic[obj_id]:
                self.container.rigid_body_centers_of_mass[obj_id] = self.rigid_body_temp_centers_of_mass[obj_id]
                