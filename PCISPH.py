import taichi as ti
import numpy as np
from pcisph_container_3d import PCISPHContainer3D

@ti.data_oriented
class PCISPHSolver3D():
    def __init__(self, container: PCISPHContainer3D):
        self.container = container
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  

        self.surface_tension = 0.01

        self.max_iterations = 20
        self.eta = 0.005

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

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

    @ti.kernel
    def compute_pcisph_k(self):
        for p_i in range(self.container.particle_num[None]):
            ret_i = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.container.for_all_neighbors(p_i, self.compute_pcisph_k_task, ret_i)
            sum_nabla = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(self.container.dim)):
                sum_nabla[i] = ret_i[i]

            volume_i = self.container.particle_reference_volumes[p_i]

            self.container.particle_pcisph_k[p_i] = (
                - 0.5 
                / (ti.math.dot(sum_nabla, sum_nabla) + ret_i[3]) 
                / (self.dt[None] * volume_i)
                / (self.dt[None] * volume_i) 
            )


    @ti.func
    def compute_pcisph_k_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.cubic_kernel_derivative(R)

        ret[0] += nabla_ij[0]
        ret[1] += nabla_ij[1]
        ret[2] += nabla_ij[2]
        ret[3] += ti.math.dot(nabla_ij, nabla_ij)
       

    @ti.kernel
    def compute_density_star(self):
        for p_i in range(self.container.particle_num[None]):
            density_i = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_star_task, density_i)
            density_i *= self.dt[None]
            density_i += self.container.particle_densities[p_i]
            self.container.particle_densities_star[p_i] = density_i

    
    @ti.func
    def compute_density_star_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        m_j = self.container.particle_masses[p_j]
        v_ij = self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]
        a_ij = self.container.particle_accelerations[p_i] - self.container.particle_accelerations[p_j]
        ret += (
            m_j * ti.math.dot(v_ij, self.cubic_kernel_derivative(R))
            + m_j * self.dt[None] * ti.math.dot(a_ij, self.cubic_kernel_derivative(R))
        )


    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_pressures[p_i] = (
                self.container.particle_pcisph_k[p_i] 
                * (self.density_0 - self.container.particle_densities_star[p_i])
            )

            if self.container.particle_pressures[p_i] < 0.0:
                self.container.particle_pressures[p_i] = 0.0

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
    def compute_pressure_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            ret_i = ti.Vector([0.0, 0.0, 0.0])
            self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, ret_i)
            self.container.particle_pressure_accelerations[p_i] = ret_i


    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        den_j = self.container.particle_densities[p_j]
        R = pos_i - pos_j
        ret += (
            - self.container.particle_masses[p_j] * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) * self.cubic_kernel_derivative(R)
        )

    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_densities[p_i] = self.container.particle_reference_volumes[p_i] * self.cubic_kernel(0.0)
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
            ret += self.container.particle_reference_volumes[p_j] * self.cubic_kernel(R_mod)

    @ti.kernel
    def compute_density_change(self):
        for p_i in range(self.container.particle_num[None]):
            density_change_i = 0.0
            self.container.for_all_neighbors(p_i, self.compute_density_change_task, density_change_i)
            density_change_i *= self.dt[None]
            self.container.particle_densities_change[p_i] = density_change_i

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.cubic_kernel_derivative(R)
        a_ij = self.container.particle_pressure_accelerations[p_i] - self.container.particle_pressure_accelerations[p_j]
        ret += self.container.particle_masses[p_j] * ti.math.dot(a_ij, nabla_ij) * self.dt[None]

    @ti.kernel
    def update_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            # ! how to treat sparse area?
            if self.container.particle_densities_star > self.density_0 + 1e-5:
                self.container.particle_pressures[p_i] += self.container.particle_pcisph_k[p_i] * (self.density_0 - self.container.particle_densities_star[p_i])
                ret = 0.0
                self.container.for_all_neighbors(p_i, self.update_pressure_task, ret)
                self.container.particle_pressures[p_i] += ret * self.dt[None]


    @ti.func
    def update_pressure_task(self, p_i, p_j, ret: ti.template()):
        a_ij = self.container.particle_pressure_accelerations[p_i] - self.container.particle_pressure_accelerations[p_j]
        R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
        nabla_ij = self.cubic_kernel_derivative(R)
        ret += (
            - self.container.particle_pcisph_k[p_i]
            * self.dt[None]
            * ti.math.dot(a_ij, nabla_ij)
            * self.container.particle_masses[p_j]
        )

    @ti.kernel
    def compute_density_error(self) -> ti.f32:
        error = 0.0
        for p_i in range(self.container.particle_num[None]):
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

        while num_itr < 1 or num_itr < self.max_iterations:
            self.compute_pressure_acceleration()
            self.compute_density_change()
            density_average_error = self.compute_density_error()
            density_average_error = ti.abs(density_average_error)

            if density_average_error < self.eta:
                break
            num_itr += 1
        # if density_average_error > 10:
        #     for p_i in range(self.container.particle_num[None]):
        #         e = (
        #             (self.density_0 - self.container.particle_densities_star[p_i] + self.container.particle_densities_change[p_i]) 
        #             / self.density_0)
        #         if ti.abs(e) > 10:
        #             print(f"particle {p_i}, density error: {e}, density: {self.container.particle_densities_star[p_i]}, density change: {self.container.particle_densities_change[p_i]}, density_star: {self.container.particle_densities_star[p_i]}")
                    
        #     breakpoint()
        print(f"Density average error ratio: {density_average_error}, num_itr: {num_itr}")

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

    @ti.kernel
    def update_velocities(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i] + self.dt[None] * self.container.particle_pressure_accelerations[p_i]

    @ti.kernel
    def update_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
    


    def step(self):
        
        self.container.prepare_neighborhood_search()
        self.compute_pcisph_k()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.compute_density_star()
        self.compute_pressure()
        self.refine()

        self.update_velocities()
        self.update_position()


        self.enforce_boundary_3D(self.container.material_fluid)