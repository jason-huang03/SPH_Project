# implementation of paper "Divergence-Free Smoothed Particle Hydrodynamics"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
import numpy as np
from ..containers import DFSPHContainer
from .base_solver import BaseSolver

@ti.data_oriented
class DFSPHSolver(BaseSolver):
    def __init__(self, container: DFSPHContainer):
        super().__init__(container)

        # dfsph related parameters
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

        self.m_eps = 1e-5

        self.max_error_V = 0.001
        self.max_error = 0.0001
    
    @ti.kernel
    def compute_alpha(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.container.dim)])
            
            ret = ti.Vector([0.0 for _ in range(self.container.dim+1)])
            
            self.container.for_all_neighbors(p_i, self.compute_alpha_task, ret)
            
            sum_grad_p_k = ret[self.container.dim]
            for i in ti.static(range(self.container.dim)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-5:
                factor = 1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.container.particle_dfsph_alphas[p_i] = factor
            
    @ti.func
    def compute_alpha_task(self, p_i, p_j, ret: ti.template()):
        # here we use partilce rest volume instead of mass
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
            ret[self.container.dim] += grad_p_j.norm_sqr() # sum_grad_p_k
            for i in ti.static(range(self.container.dim)): # grad_p_i
                ret[i] += grad_p_j[i]
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # Rigid neighbors
            # we discard the square norm term
            grad_p_j = -self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
            for i in ti.static(range(self.container.dim)):
                ret[i] += grad_p_j[i]
    
    ################# Density Related Computation #################
    @ti.kernel
    def compute_density_derivative(self):
        """
        compute (D rho / Dt) / rho_0 for each particle
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Struct(density_adv=0.0, num_neighbors=0)
                self.container.for_all_neighbors(p_i, self.compute_density_derivative_task, ret)

                # only correct positive divergence
                density_adv = ti.max(ret.density_adv, 0.0)
                num_neighbors = ret.num_neighbors

                # Do not perform divergence solve when paritlce deficiency happens
                if self.container.dim == 3:
                    if num_neighbors < 20:
                        density_adv = 0.0
                else:
                    if num_neighbors < 7:
                        density_adv = 0.0
        
                self.container.particle_densities_derivatives[p_i] = density_adv


    @ti.func
    def compute_density_derivative_task(self, p_i, p_j, ret: ti.template()):
        # here we use partilce rest volume instead of mass
        # Fluid neighbor and rigid neighbor are treated the same
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        ret.density_adv += self.container.particle_rest_volumes[p_j] * ti.math.dot(v_i - v_j, self.kernel_gradient(pos_i - pos_j))
 
        # Compute the number of neighbors
        ret.num_neighbors += 1
    

    @ti.kernel
    def compute_density_star(self):
        """
        compute rho^* / rho_0 for each particle
        """
        for p_i in range(self.container.particle_num[None]):
            delta = 0.0
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.for_all_neighbors(p_i, self.compute_density_star_task, delta)
                density_adv = self.container.particle_densities[p_i] /self.density_0 + self.dt[None] * delta
                self.container.particle_densities_star[p_i] = ti.max(density_adv, 1.0)


    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        # here we use partilce rest volume instead of mass
        # Fluid neighbor and rigid neighbor are treated the same
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
            
        ret += self.container.particle_rest_volumes[p_j] * ti.math.dot(v_i - v_j,self.kernel_gradient(pos_i - pos_j))

    ################# End of Density Related Computation #################


    ################# Divergence Free Solver #################
    @ti.kernel
    def compute_kappa_v(self):
        # note we do not divide delta_t here
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                self.container.particle_dfsph_kappa_v[idx_i] = self.container.particle_densities_derivatives[idx_i] * self.container.particle_dfsph_alphas[idx_i] 
    
    def correct_divergence_error(self):
        num_itr = 0
        self.compute_density_derivative()
        average_density_derivative_error = 0.0

        while num_itr < 1 or num_itr < self.m_max_iterations_v:
            
            self.compute_kappa_v()
            self.correct_divergence_step()
            self.compute_density_derivative()
            average_density_derivative_error = self.compute_density_derivative_error()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = self.max_error_V * self.density_0 / self.dt[None]
            num_itr += 1

            if average_density_derivative_error <= eta:
                break

        
        print(f"DFSPH - iteration V: {num_itr} Avg density err: {average_density_derivative_error}")

    @ti.kernel
    def correct_divergence_step(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                k_i = self.container.particle_dfsph_kappa_v[p_i]
                ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.container.dim)]), k_i=k_i)
        
                self.container.for_all_neighbors(p_i, self.correct_divergence_task, ret)
                self.container.particle_velocities[p_i] += ret.dv
            
    
    @ti.func
    def correct_divergence_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            k_j = self.container.particle_dfsph_kappa_v[p_j]
            k_i = ret.k_i
            k_sum = k_i + k_j 
            if ti.abs(k_sum) > self.m_eps * self.dt[None]:
                grad_p_j = self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                ret.dv -= grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            k_i = ret.k_i
            k_j = k_i
            # ? how to implement here?
            k_sum = k_i
            den_i = self.container.particle_densities[p_i]
            if ti.abs(k_sum) > self.m_eps * self.dt[None]:
                grad_p_j = self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                ret.dv -= grad_p_j * (k_i / den_i) * self.density_0
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                    force_j = (
                        grad_p_j * (k_i / den_i) * self.density_0 / self.dt[None] 
                        * (self.container.particle_rest_volumes[p_i] * self.density_0)
                    )
                    torque_j = ti.math.cross(self.container.particle_positions[p_j] - center_of_mass_j, force_j)
                    self.container.rigid_body_forces[object_j] += force_j
                    self.container.rigid_body_torques[object_j] += torque_j


    @ti.kernel
    def compute_density_derivative_error(self) -> float:
        density_error = 0.0
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.density_0 * self.container.particle_densities_derivatives[idx_i]
        return density_error / self.container.particle_num[None]
    
    ################# End of Divergence Free Solver #################


    ################# Constant Density Solver #################
    @ti.kernel
    def compute_kappa(self):
        # note we only divide delta_t once here
        delta_t_inv = 1 / (self.dt[None])
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                self.container.particle_dfsph_kappa[idx_i] = (self.container.particle_densities_star[idx_i] - 1.0) * self.container.particle_dfsph_alphas[idx_i] * delta_t_inv

    def correct_density_error(self):
        self.compute_density_star()
        num_itr = 0

        average_density_error = 0.0

        while num_itr < 1 or num_itr < self.m_max_iterations:
            
            self.compute_kappa()
            self.correct_density_error_step()
            self.compute_density_star()
            average_density_error = self.compute_density_error()
            # Max allowed density fluctuation
            eta = self.max_error
            num_itr += 1

            if average_density_error <= eta:
                break
        print(f"DFSPH - iterations: {num_itr} Avg density Err: {average_density_error * self.density_0:.4f}")

    @ti.kernel
    def correct_density_error_step(self):
        # Compute pressure forces
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                k_i = self.container.particle_dfsph_kappa[p_i]
                self.container.for_all_neighbors(p_i, self.correct_density_error_task, k_i)
        

    @ti.func
    def correct_density_error_task(self, p_i, p_j, k_i: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            k_j = self.container.particle_dfsph_kappa[p_j]
            k_sum = k_i +  k_j 
            if ti.abs(k_sum) > self.m_eps * self.dt[None]:
                grad_p_j = self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                self.container.particle_velocities[p_i] -= grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # Rigid neighbors
            k_j = k_i
            den_i = self.container.particle_densities[p_i]
            # ? how to implement here?
            k_sum = k_i
            if ti.abs(k_sum) > self.m_eps * self.dt[None]:
                grad_p_j = self.container.particle_rest_volumes[p_j] * self.kernel_gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                self.container.particle_velocities[p_i] -= grad_p_j * (k_i / den_i) * self.density_0
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                    force_j = (
                        grad_p_j * (k_i / den_i) * self.density_0 / self.dt[None]
                        * (self.container.particle_rest_volumes[p_i] * self.density_0)
                    )
                    torque_j = ti.math.cross(self.container.particle_positions[p_j] - center_of_mass_j, force_j)
                    self.container.rigid_body_forces[object_j] += force_j
                    self.container.rigid_body_torques[object_j] += torque_j

    @ti.kernel
    def compute_density_error(self) -> float:
        """
        compute average of (rho^* / rho_0) - 1
        """
        density_error = 0.0
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.container.particle_densities_star[idx_i] - 1.0
        return density_error / self.container.particle_num[None]

    ################# End of Constant Density Solver #################

    def _step(self):
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()
        self.correct_density_error()

        self.update_fluid_position()

        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()

        if self.container.dim == 3:
            self.enforce_domain_boundary_3D(self.container.material_fluid)
        else:
            self.enforce_domain_boundary_2D(self.container.material_fluid)

        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_alpha()
        self.correct_divergence_error()

    def prepare(self):
        super().prepare()
        self.compute_density()
        self.compute_alpha()
