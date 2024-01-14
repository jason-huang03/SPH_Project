# implementation of paper "Divergence-Free Smoothed Particle Hydrodynamics"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
import numpy as np
from ..containers import IISPHContainer
from .base_solver import BaseSolver

@ti.data_oriented
class IISPHSolver(BaseSolver):
    def __init__(self, container: IISPHContainer):
        super().__init__(container)
        self.max_iterations = 20
        self.eta = 0.001
        self.omega = 0.2


    @ti.kernel
    def compute_dii(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(p_i, self.compute_dii_task, ret)
                self.container.dii[p_i] = ret

    @ti.func
    def compute_dii_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            ret += (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * nabla_ij 
                / self.container.particle_densities[p_j] / self.container.particle_densities[p_j]
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            ret += (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * nabla_ij 
                / self.container.particle_densities_star[p_i] / self.container.particle_densities_star[p_i]
            )

    @ti.kernel
    def compute_aii(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(p_i, self.compute_aii_task, ret)
                self.container.iisph_aii[p_i] = ret * self.dt[None] * self.dt[None]

    @ti.func
    def compute_aii_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        dii = self.container.dii[p_i]
        dji = self.density_0 * self.container.particle_rest_volumes[p_i] * nabla_ij / self.container.particle_densities[p_i] / self.container.particle_densities[p_i]

        # treat rigid neighbors and fluid neighbors the same
        ret += (
            self.density_0 * self.container.particle_rest_volumes[p_j]
            * ti.math.dot(dii - dji, nabla_ij)
        )

    @ti.kernel
    def compute_density_star(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_star_task, ret)
                self.container.particle_densities_star[p_i] = self.container.particle_densities[p_i] + self.dt[None] * ret


    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        # here we use partilce rest volume instead of mass
        # Fluid neighbor and rigid neighbor are treated the same
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)
            
        ret += self.density_0 * self.container.particle_rest_volumes[p_j] * ti.math.dot(v_i - v_j, nabla_ij)

    @ti.kernel
    def init_step(self):
        self.container.particle_pressures.fill(0.0)


    @ti.kernel
    def update_pressure(self):
        error = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                new_pressure = 0.0
                si = self.density_0 - self.container.particle_densities_star[p_i]
                if self.container.iisph_aii[p_i] > 1e-10 or self.container.iisph_aii[p_i] < -1e-10:
                    new_pressure = (
                        (1 - self.omega) * self.container.particle_pressures[p_i]
                        + self.omega / self.container.iisph_aii[p_i] * (si - self.container.sum_i[p_i])
                    )

                    new_pressure = ti.max(0.0, new_pressure)

                self.container.particle_pressures[p_i] = new_pressure

                if new_pressure > 1e-10:
                    error += (
                        self.container.iisph_aii[p_i] * new_pressure + self.container.sum_i[p_i] - si
                    )

        if self.container.fluid_particle_num[None] > 0:
            self.container.density_error[None] = error / self.container.fluid_particle_num[None] / self.density_0
        else:
            self.container.density_error[None] = 0.0

    @ti.kernel
    def compute_dij_pj(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_dij_pj_task, ret)
                self.container.dij_pj[p_i] = ret

    @ti.func
    def compute_dij_pj_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            pos_i = self.container.particle_positions[p_i]
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            nabla_ij = self.kernel_gradient(R)

            ret += (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * nabla_ij 
                / self.container.particle_densities[p_j] / self.container.particle_densities[p_j]
                * self.container.particle_pressures[p_j]
            )

    @ti.kernel
    def compute_sum_i(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(p_i, self.compute_sum_i_task, ret)
                self.container.sum_i[p_i] = ret * self.dt[None] * self.dt[None]
    
    @ti.func
    def compute_sum_i_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        dpi = self.density_0 * self.container.particle_rest_volumes[p_i] / self.container.particle_densities[p_i] / self.container.particle_densities[p_i]

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            d_jk_pk = self.container.dij_pj[p_j]
            dji = dpi * nabla_ij
            d_ji_pi = dji * self.container.particle_pressures[p_i]

            temp = (
                self.density_0 * self.container.particle_rest_volumes[p_j] 
                * (
                    self.container.dij_pj[p_i] - self.container.dii[p_j] * self.container.particle_pressures[p_j] - (d_jk_pk - d_ji_pi)
                )
            )
            ret += ti.math.dot(temp, nabla_ij)
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            temp = (
                self.density_0 * self.container.particle_rest_volumes[p_j]
                * self.container.dij_pj[p_i]
            )
            ret += ti.math.dot(temp, nabla_ij)
            

    def refine(self):
        num_itr = 0

        while num_itr < self.max_iterations:
            self.compute_dij_pj()
            self.compute_sum_i()
            self.update_pressure()

            num_itr += 1



            if self.container.density_error[None] < self.eta:
                break

        print(f"IISPH - iteration: {num_itr} Avg density err: {self.container.density_error[None] * self.density_0}")


    def _step(self):
        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.init_step()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()

        self.compute_dii()
        self.compute_aii()
        self.compute_density_star()

        self.refine()

        # same procedure as WCSPH
        # use pressure to compute acceleration for fluid and rigid body
        self.compute_pressure_acceleration()
        self.update_fluid_velocity()
        self.update_fluid_position()

        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()

        self.enforce_domain_boundary_3D(self.container.material_fluid)