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
        self.max_iterations = 1000
        self.eta = 0.001


    @ti.kernel
    def compute_predicted_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_predicted_velocities[p_i] = self.container.particle_velocities[p_i] + self.dt[None] * (self.container.particle_accelerations[p_i] + self.container.particle_pressure_accelerations[p_i])


    @ti.kernel
    def compute_predicted_position(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_predicted_positions[p_i] = self.container.particle_positions[p_i] + self.dt[None] * self.container.particle_predicted_velocities[p_i]


    @ti.kernel
    def compute_density_star(self):
        error = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_star_task, ret)
                self.container.particle_densities_star[p_i] = ret * self.density_0
                error += ti.max(0.0, ret - 1.0)

        if self.container.fluid_particle_num[None] > 0:
            self.container.density_error[None] = error / self.container.fluid_particle_num[None]
        else:
            self.container.density_error[None] = 0.0    
                

    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_predicted_positions[p_i]

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            pos_j = self.container.particle_predicted_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.container.particle_rest_volumes[p_j] * self.kernel_W(R_mod)
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.container.particle_rest_volumes[p_j] * self.kernel_W(R_mod)


    @ti.kernel
    def update_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_pressures[p_i] += self.container.pcisph_k[None] * (self.density_0 - self.container.particle_densities_star[p_i])
                if self.container.particle_pressures[p_i] < 0.0:
                    self.container.particle_pressures[p_i] = 0.0

    
    @ti.kernel
    def compute_temp_pressure_acceleration(self):
        self.container.particle_pressure_accelerations.fill(0.0)
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_temp_pressure_acceleration_task, ret_i)
                self.container.particle_pressure_accelerations[p_i] = ret_i
                

    @ti.func
    def compute_temp_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_i = self.container.particle_densities[p_i]
            den_j = self.container.particle_densities[p_j]

            ret += (
                - self.container.particle_masses[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) 
                * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            den_i = self.container.particle_densities[p_i]

            ret += (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i)) * nabla_ij
            )


    def refine(self):
        num_itr = 0

        while num_itr < self.max_iterations:
            self.compute_density_star()
            self.update_pressure()
            self.compute_temp_pressure_acceleration()
            self.compute_predicted_velocity()
            self.compute_predicted_position()

            num_itr += 1

            if self.container.density_error[None] < self.eta:
                break
            
        print(f"PCISPH - iteration: {num_itr} Avg density err: {self.container.density_error[None] * self.density_0}")


    @ti.kernel
    def compute_pcisph_k(self):
        # if using adaptive time step, this function needed to be called every step
        support_radius = self.container.dh
        diam = self.container.particle_diameter * 0.97
        pos_i = ti.Vector([0.0, 0.0, 0.0])

        sumGradW = ti.Vector([0.0, 0.0, 0.0])
        sumGradW2 = 0.0

        # iterate over every points in cube [-support_radius, support_radius]^3, spaced at diam
        max_i = int(support_radius / diam) + 1

        for i in range(-max_i, max_i + 1):
            for j in range(-max_i, max_i + 1):
                for k in range(-max_i, max_i + 1):
                    pos_j = ti.Vector([i * diam, j * diam, k * diam])
                    x_ij = pos_i - pos_j
                    if x_ij.norm() < support_radius:
                        nabla_ij = self.kernel_gradient(x_ij)
                        sumGradW += nabla_ij
                        sumGradW2 += nabla_ij.norm_sqr()

        self.container.pcisph_k[None] = - 0.5 / (self.dt[None] * self.container.V0) / (self.dt[None] * self.container.V0) / (sumGradW.norm_sqr() + sumGradW2)

    @ti.kernel
    def init_step(self):
        self.container.particle_pressure_accelerations.fill(0.0)
        self.container.particle_pressures.fill(0.0)
        self.container.density_error[None] = 100.0

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_predicted_velocities[p_i] = self.container.particle_velocities[p_i] + self.dt[None] * self.container.particle_accelerations[p_i]
                self.container.particle_predicted_positions[p_i] = self.container.particle_positions[p_i] + self.dt[None] * self.container.particle_predicted_velocities[p_i]


    def _step(self):
        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.init_step()
        self.refine() # compute correct pressure here

        # same procedure as WCSPH
        # use pressure to compute acceleration for fluid and rigid body
        self.update_fluid_velocity()
        self.compute_pressure_acceleration()
        self.update_fluid_velocity()
        self.update_fluid_position()

        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()

        self.enforce_domain_boundary_3D(self.container.material_fluid)


    def prepare(self):
        super().prepare()
        self.compute_pcisph_k()