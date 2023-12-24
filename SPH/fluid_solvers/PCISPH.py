# implementatioin of paper "Predictive-Corrective Incompressible SPH"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
import numpy as np
from ..containers import PCISPHContainer
from .base_solver import BaseSolver

@ti.data_oriented
class PCISPHSolver(BaseSolver):
    def __init__(self, container: PCISPHContainer):
        super().__init__(container)

        # pcisph related parameters
        self.max_iterations = 50
        self.eta = 0.005

    @ti.kernel
    def compute_pcisph_k(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret_i = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.container.for_all_neighbors(p_i, self.compute_pcisph_k_task, ret_i)
                sum_nabla = ti.Vector([0.0, 0.0, 0.0])
                for i in ti.static(range(self.container.dim)):
                    sum_nabla[i] = ret_i[i]

                volume_i = self.container.particle_rest_volumes[p_i]

                self.container.particle_pcisph_k[p_i] = (
                    - 0.5 
                    / (ti.math.dot(sum_nabla, sum_nabla) + ret_i[3]) 
                    / (self.dt[None] * volume_i)
                    / (self.dt[None] * volume_i) 
                )


    @ti.func
    def compute_pcisph_k_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbor and rigid neighbor are treated the same here
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        ret[0] += nabla_ij[0]
        ret[1] += nabla_ij[1]
        ret[2] += nabla_ij[2]
        ret[3] += ti.math.dot(nabla_ij, nabla_ij)
       

    @ti.kernel
    def compute_density_star(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                density_i = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_star_task, density_i)
                density_i *= self.dt[None]
                density_i += self.container.particle_densities[p_i]
                self.container.particle_densities_star[p_i] = density_i

    
    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbor and rigid neighbor are treated the same here
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)
        v_ij = self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]
        m_j = self.container.particle_rest_volumes[p_j] * self.density_0
        a_ij = self.container.particle_accelerations[p_i] - self.container.particle_accelerations[p_j]
        ret += (
            m_j * ti.math.dot(v_ij, nabla_ij)
            + m_j * self.dt[None] * ti.math.dot(a_ij, nabla_ij)
        )


    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_pressures[p_i] = (
                    self.container.particle_pcisph_k[p_i] 
                    * (self.density_0 - self.container.particle_densities_star[p_i])
                )

                if self.container.particle_pressures[p_i] < 0.0:
                    self.container.particle_pressures[p_i] = 0.0



    @ti.kernel
    def update_pressure_acceleration(self):
        self.container.rigid_body_pressure_forces.fill(0.0)
        self.container.rigid_body_pressure_torques.fill(0.0)
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret_i = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(p_i, self.update_pressure_acceleration_task, ret_i)
                self.container.particle_pressure_accelerations[p_i] = ret_i


    @ti.func
    def update_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        R = pos_i - self.container.particle_positions[p_j]
        nabla_ij = self.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_i = self.container.particle_densities[p_i]
            den_j = self.container.particle_densities[p_j]

            ret += (
                - self.container.particle_rest_volumes[p_j] * self.density_0 * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            den_i = self.container.particle_densities[p_i]
            den_j = den_i
            pressure_i = self.container.particle_pressures[p_i]

            ret += (
                - self.density_0 * self.container.particle_rest_volumes[p_j] * (pressure_i / (den_i * den_i)) * nabla_ij
            )

            if self.container.particle_is_dynamic[p_j]:
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j = (
                    self.density_0 * self.container.particle_rest_volumes[p_j] * pressure_i / (den_i * den_i) * nabla_ij
                    * (self.density_0 * self.container.particle_rest_volumes[p_i])
                )

                torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
                self.container.rigid_body_pressure_forces[object_j] += force_j
                self.container.rigid_body_pressure_torques[object_j] += torque_j




    @ti.kernel
    def compute_density_change(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                density_change_i = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_change_task, density_change_i)
                density_change_i *= self.dt[None]
                self.container.particle_densities_change[p_i] = density_change_i

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbor and rigid neighbor are treated the same here
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)
        a_ij = self.container.particle_pressure_accelerations[p_i] - self.container.particle_pressure_accelerations[p_j]
        ret += self.container.particle_rest_volumes[p_j] * self.density_0 * ti.math.dot(a_ij, nabla_ij) * self.dt[None]


    @ti.kernel
    def update_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            # ! how to treat sparse area?
            if self.container.particle_materials[p_i] == self.container.material_fluid: 
                if self.container.particle_densities_star[p_i] > self.density_0 + 1e-5:
                    self.container.particle_pressures[p_i] += self.container.particle_pcisph_k[p_i] * (self.density_0 - self.container.particle_densities_star[p_i])
                    ret = 0.0
                    self.container.for_all_neighbors(p_i, self.update_pressure_task, ret)
                    self.container.particle_pressures[p_i] += ret * self.dt[None]


    @ti.func
    def update_pressure_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbor and rigid neighbor are treated the same here
        a_ij = self.container.particle_pressure_accelerations[p_i] - self.container.particle_pressure_accelerations[p_j]
        R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
        nabla_ij = self.kernel_gradient(R)
        ret += (
            - self.container.particle_pcisph_k[p_i]
            * self.dt[None]
            * ti.math.dot(a_ij, nabla_ij)
            * self.container.particle_rest_volumes[p_j] * self.density_0
        )

    @ti.kernel
    def compute_density_error(self) -> ti.f32:
        error = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                if self.container.particle_densities_star[p_i] > self.density_0 + 1e-5:
                    error += (
                        (self.density_0 - self.container.particle_densities_star[p_i] + self.container.particle_densities_change[p_i]) 
                        / self.density_0
                    )

        error = error / self.container.particle_num[None]
        return error

    def refine(self):
        num_itr = 0
        density_average_error = 0.0

        while num_itr < 2 or num_itr < self.max_iterations:
            
            self.update_pressure_acceleration()
            self.compute_density_change()
            self.update_pressure()

            density_average_error = self.compute_density_error()
            density_average_error = ti.abs(density_average_error)

            if density_average_error < self.eta:
                break
            num_itr += 1

        print(f"Density average error ratio: {density_average_error}, num_itr: {num_itr}")


    @ti.kernel
    def update_fluid_velocity(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i] + self.dt[None] * self.container.particle_pressure_accelerations[p_i]

    @ti.kernel
    def update_fluid_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
    
    @ti.kernel
    def apply_pressure_force_to_rigid(self):
        for object_i in range(self.container.object_num[None]):
            if self.container.particle_is_dynamic[object_i] and self.container.object_materials[object_i] == self.container.material_rigid:
                self.container.rigid_body_forces[object_i] += self.container.rigid_body_pressure_forces[object_i]
                self.container.rigid_body_torques[object_i] += self.container.rigid_body_pressure_torques[object_i]
            
    def _step(self):
        self.container.prepare_neighborhood_search()
        self.init_acceleration()
        self.compute_pcisph_k()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.compute_density_star()
        self.compute_pressure()
        self.refine()

        self.update_fluid_velocity()
        self.update_fluid_position()

        self.apply_pressure_force_to_rigid()
        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()

        self.enforce_boundary_3D(self.container.material_fluid)