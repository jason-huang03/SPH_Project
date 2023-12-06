import taichi as ti
import numpy as np
from dfsph_container_3d import DFSPHContainer3D

@ti.data_oriented
class DFSPHSolver3D():
    def __init__(self, container: DFSPHContainer3D):
        self.container = container
        if self.container.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        elif self.container.dim == 3:
            self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  

        self.surface_tension = 0.01

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05
    

    ################# kernel #################
    @ti.func
    def cubic_kernel(self, R_mod):
        res = ti.cast(0.0, ti.f32)
        h = self.container.dh
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k /= h ** self.container.dim
        q = R_mod / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, R):
        h = self.container.dh
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.container.dim
        R_mod = R.norm()
        q = R_mod / h
        res = ti.Vector([0.0 for _ in range(self.container.dim)])
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    ################# End of kernel #################

    @ti.kernel
    def compute_non_pressure_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            ############## Body force ###############
            # Add body force
            a_i = ti.Vector(self.g)
            self.container.particle_accelerations[p_i] = a_i
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.for_all_neighbors(p_i, self.compute_non_pressure_acceleration_task, a_i)
                self.container.particle_accelerations[p_i] = a_i


    @ti.func
    def compute_non_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        
        ############## Surface Tension ###############
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            diameter2 = self.container.particle_diameter * self.container.particle_diameter
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = ti.math.dot(R, R)
            if R2 > diameter2:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.cubic_kernel(R.norm())
            else:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.cubic_kernel(ti.Vector([self.container.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.container.dim + 2)
        pos_j = self.container.particle_positions[p_j]
        # Compute the viscosity force contribution
        R = pos_i - pos_j
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)
        
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            f_v = d * self.viscosity * (self.container.particle_masses[p_j] / (self.container.particle_densities[p_j])) * v_xy / (
                R.norm()**2 + 0.01 * self.container.dh**2) * self.cubic_kernel_derivative(R)
            ret += f_v
  


    @ti.kernel
    def compute_alpha(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.container.dim)])
            
            ret = ti.Vector([0.0, 0.0, 0.0, 0.0])
            
            self.container.for_all_neighbors(p_i, self.compute_alpha_task, ret)
            
            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.container.particle_dfsph_alphas[p_i] = factor
            

    @ti.func
    def compute_alpha_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.container.particle_original_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
            ret[3] += grad_p_j.norm_sqr() # sum_grad_p_k
            for i in ti.static(range(3)): # grad_p_i
                ret[i] += grad_p_j[i]
    
    ################# Density Related Computation #################
    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_densities[p_i] = self.container.particle_original_volumes[p_i] * self.cubic_kernel(0.0)
            density_i = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_task, density_i)
            self.container.particle_densities[p_i] += density_i
            self.container.particle_densities[p_i] *= self.density_0

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.container.particle_original_volumes[p_j] * self.cubic_kernel(R_mod)

    @ti.kernel
    def compute_density_derivative(self):
        """
        compute (D rho / Dt) / rho_0 for each particle
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
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
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        ret.density_adv += self.container.particle_original_volumes[p_j] * ti.math.dot(v_i - v_j, self.cubic_kernel_derivative(pos_i - pos_j))
 
        # Compute the number of neighbors
        ret.num_neighbors += 1
    

    @ti.kernel
    def compute_density_star(self):
        """
        compute rho^* / rho_0 for each particle
        """
        for p_i in range(self.container.particle_num[None]):
            delta = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_star_task, delta)
            density_adv = self.container.particle_densities[p_i] /self.density_0 + self.dt[None] * delta
            self.container.particle_densities_star[p_i] = ti.max(density_adv, 1.0)


    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            ret += self.container.particle_original_volumes[p_j] * ti.math.dot(v_i - v_j,self.cubic_kernel_derivative(pos_i - pos_j))

    ################# End of Density Related Computation #################


    ################# Divergence Free Solver #################
    @ti.kernel
    def compute_kappa_v(self):
        for idx_i in range(self.container.particle_num[None]):
            self.container.particle_dfsph_kappa_v[idx_i] = self.container.particle_densities_derivatives[idx_i] * self.container.particle_dfsph_alphas[idx_i] 

            
    def correct_divergence_error(self):
        # TODO: warm start 
        self.compute_density_derivative()

        num_itr = 0
        
        average_density_derivative_error = 0.0

        while num_itr < 1 or num_itr < self.m_max_iterations_v:
            
            self.compute_kappa_v()
            self.correct_divergence_step()
            self.compute_density_derivative()
            average_density_derivative_error = self.compute_density_derivative_error()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = 1.0 / self.dt[None] * self.max_error_V * 0.01 * self.density_0

            if average_density_derivative_error <= eta:
                break
            num_itr += 1
        print(f"DFSPH - iteration V: {num_itr} Avg density err: {average_density_derivative_error}")

    @ti.kernel
    def correct_divergence_step(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
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
                grad_p_j = -self.container.particle_original_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                ret.dv -= grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0

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
        delta_t_inv = 1 / (self.dt[None])
        for idx_i in range(self.container.particle_num[None]):
            self.container.particle_dfsph_kappa[idx_i] = (self.container.particle_densities_star[idx_i] - 1.0) * self.container.particle_dfsph_alphas[idx_i] * delta_t_inv

    def correct_density_error(self):
        # TODO: warm start
        self.compute_density_star()
        num_itr = 0

        average_density_error = 0.0

        while num_itr < 1 or num_itr < self.m_max_iterations:
            
            self.compute_kappa()
            self.correct_density_error_step()
            self.compute_density_star()
            average_density_error = self.compute_density_error()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01
            if average_density_error <= eta:
                break
            num_itr += 1
        print(f"DFSPH - iterations: {num_itr} Avg density Err: {average_density_error * self.density_0:.4f}")
        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

    
    @ti.kernel
    def correct_density_error_step(self):
        # Compute pressure forces
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_fluid:
                continue
            k_i = self.container.particle_dfsph_kappa[p_i]

            self.container.for_all_neighbors(p_i, self.correct_density_error_task, k_i)
    

    @ti.func
    def correct_density_error_task(self, p_i, p_j, k_i: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            k_j = self.container.particle_dfsph_kappa[p_j]
            k_sum = k_i +  k_j 
            if ti.abs(k_sum) > self.m_eps * self.dt[None]:
                grad_p_j = -self.container.particle_original_volumes[p_j] * self.cubic_kernel_derivative(self.container.particle_positions[p_i] - self.container.particle_positions[p_j])
                # Directly update velocities instead of storing pressure accelerations
                self.container.particle_velocities[p_i] -= grad_p_j * (k_i / self.container.particle_densities[p_i] + k_j / self.container.particle_densities[p_j]) * self.density_0

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

    ################# Enforce Boundary #################
    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.container.particle_velocities[p_i] -= (
            1.0 + c_f) * self.container.particle_velocities[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]:
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding

                if pos[2] > self.container.domain_size[2] - self.container.padding:
                    collision_normal[2] += 1.0
                    self.container.particle_positions[p_i][2] = self.container.domain_size[2] - self.container.padding
                if pos[2] <= self.container.padding:
                    collision_normal[2] += -1.0
                    self.container.particle_positions[p_i][2] = self.container.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

    ################# End of Enforce Boundary #################

    @ti.kernel
    def update_velocities(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
    

    def step(self):

        self.compute_non_pressure_acceleration()
        self.update_velocities()
        self.correct_density_error()
        self.update_position()
        self.enforce_boundary_3D(self.container.material_fluid)

        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_alpha()
        self.correct_divergence_error()