# implementatioin of paper "Position Based Fluids"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
import numpy as np
from ..containers import PBFContainer
from .base_solver import BaseSolver

@ti.data_oriented
class PBFSolver(BaseSolver):
    def __init__(self, container: PBFContainer):
        super().__init__(container)
        self.lambda_eps = 100.0
        self.corrK = 0.001
        self.corr_deltaQ_coeff = 0.3

        self.poly6d_fac = 315.0 / 64.0 / np.pi
        self.spiky_grad_fac = -45.0 / np.pi



    @ti.func
    def kernel_W(self, R_mod):
        # poly6 kernel
        h = self.container.dh
        res = 0.0

        if R_mod > 0 and R_mod < h:
            x = (h * h - R_mod * R_mod) / (h * h * h)
            res = self.poly6d_fac * x * x * x

        return res


    
    @ti.func
    def kernel_gradient(self, R):
        # spiky gradient
        R_mod = R.norm()
        h = self.container.dh

        res = ti.Vector([0.0 for _ in range(self.container.dim)])

        if R_mod > 0 and R_mod < h:
            x = (h - R_mod) / (h * h * h)
            res = self.spiky_grad_fac * x * x * R / R_mod

        return res


    @ti.func
    def compute_scorr(self, R_mod):
        x = self.kernel_W(R_mod) / self.kernel_W(self.corr_deltaQ_coeff * self.container.dh)
        # raise x to the power of 4
        x = x * x
        x = x * x

        return -self.corrK * x



    def refine(self):
        for i in range(5):
            self.compute_density()
            self.compute_lambda()
            self.fix_position()


    @ti.kernel
    def compute_lambda(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim+1)])
                self.container.for_all_neighbors(p_i, self.compute_sum_gradient_sqr_task, ret)
                sum_gradient_i = ti.Vector([ret[d] for d in range(self.container.dim)])
                sum_gradient_sqr_i = ret[self.container.dim] + sum_gradient_i.norm_sqr() + self.lambda_eps

                # ! we did not use ti.max here
                density_constraint_i = self.container.particle_densities[p_i] / self.density_0 - 1.0
                self.container.particle_pbf_lambdas[p_i] = - density_constraint_i / sum_gradient_sqr_i
                

    @ti.func
    def compute_sum_gradient_sqr_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            nabla_ij *= (self.container.particle_masses[p_j] / self.density_0)
            ret[self.container.dim] += nabla_ij.norm_sqr()
            for d in ti.static(range(self.container.dim)):
                ret[d] += nabla_ij[d]

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            den_i = self.container.particle_densities[p_i]
            den_j = den_i
            nabla_ij *= (self.container.particle_rest_volumes[p_j] * den_j / self.density_0)
            ret[self.container.dim] += nabla_ij.norm_sqr()
            for d in ti.static(range(self.container.dim)):
                ret[d] += nabla_ij[d]


    @ti.kernel
    def fix_position(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                pos_delta_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.fix_position_task, pos_delta_i)
                pos_delta_i /= self.density_0
                self.container.particle_positions[p_i] += pos_delta_i

    @ti.func
    def fix_position_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        R_mod = R.norm()
        nabla_ij = self.kernel_gradient(R)
        scorr_ij = self.compute_scorr(R_mod)

        lambda_i = self.container.particle_pbf_lambdas[p_i]

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            lambda_j = self.container.particle_pbf_lambdas[p_j]
            ret += (lambda_i + lambda_j + scorr_ij) * nabla_ij * self.container.particle_masses[p_j]

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            lambda_j = lambda_i
            den_i = self.container.particle_densities[p_i]
            ret += (lambda_i + lambda_j + scorr_ij) * nabla_ij * self.container.particle_rest_volumes[p_j] * self.density_0

    @ti.kernel
    def recompute_fluid_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] = (self.container.particle_positions[p_i] - self.container.particle_old_positions[p_i]) / self.dt[None]
                 
    @ti.kernel
    def save_old_position(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_old_positions[p_i] = self.container.particle_positions[p_i]

    def _step(self):
        self.container.prepare_neighborhood_search()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()
        self.save_old_position()
        self.update_fluid_position()

        self.enforce_domain_boundary(self.container.material_fluid)

        self.refine()

        self.enforce_domain_boundary(self.container.material_fluid)

        self.recompute_fluid_velocity()


    

       