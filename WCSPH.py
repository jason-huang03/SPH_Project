import taichi as ti
import numpy as np
from wcsph_container_3d import WCSPHContainer3D

@ti.data_oriented
class WCSPHSolver3D():
    def __init__(self, container: WCSPHContainer3D):
        self.container = container
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  

        self.surface_tension = 0.01

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        self.gamma = 7.0
        self.stiffness = 50000.0



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
    def compute_rigid_particle_volume(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                ret = self.cubic_kernel(0.0)
                self.container.for_all_neighbors(p_i, self.compute_rigid_particle_volumn_task, ret)
                self.container.particle_reference_volumes[p_i] = 1.0 / ret * 3.0 # TODO: remove this when having better samplinig method.
                self.container.particle_masses[p_i] = self.density_0 * self.container.particle_reference_volumes[p_i]


    @ti.func
    def compute_rigid_particle_volumn_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.cubic_kernel(R_mod)

    @ti.kernel
    def compute_non_pressure_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_is_dynamic[p_i]:
                self.container.particle_accelerations[p_i] = ti.Vector(self.g)
                if self.container.particle_materials[p_i] == self.container.material_fluid:
                    a_i = ti.Vector([0.0, 0.0, 0.0])
                    self.container.for_all_neighbors(p_i, self.compute_non_pressure_acceleration_task, a_i)
                    self.container.particle_accelerations[p_i] += a_i


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
        nabla_ij = self.cubic_kernel_derivative(R)
        
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            f_v = d * self.viscosity * (self.container.particle_masses[p_j] / (self.container.particle_densities[p_j])) * v_xy / (
                R.norm()**2 + 0.01 * self.container.dh**2) * nabla_ij
            ret += f_v
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # define the numerical propogation c_s
            c_s = 10.0
            nu = self.viscosity * self.container.dh * c_s / 2 / self.container.particle_densities[p_i]
            PI = - nu * ti.min(0.0, v_xy) / (R.norm_sqr() + 0.01 * self.container.dh**2)
            acc = - self.density_0 * self.container.particle_reference_volumes[p_j] * PI * nabla_ij
            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                # add force to dynamic rigid body from fluid here. 
                self.container.particle_accelerations[p_j] -= (acc * self.container.particle_masses[p_i] / self.density_0 / self.container.particle_reference_volumes[p_j])

  



    @ti.kernel
    def compute_pressure_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret_i = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, ret_i)
                self.container.particle_accelerations[p_i] += ret_i


    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        R = pos_i - pos_j

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_j = self.container.particle_densities[p_j]

            ret += (
                - self.container.particle_masses[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) 
                * self.cubic_kernel_derivative(R)
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # use fluid particle pressure, density as rigid particle pressure, density
            den_j = self.container.particle_densities[p_i]
            acc = (
                - self.density_0 * self.container.particle_reference_volumes[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_i] / (den_j * den_j)) * self.cubic_kernel_derivative(R)
            )
            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                # add force to dynamic rigid body from fluid here. 
                self.container.particle_accelerations[p_j] -= (acc * self.container.particle_masses[p_i] / self.density_0 / self.container.particle_reference_volumes[p_j])
            

    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                rho_i = self.container.particle_densities[p_i]
                rho_i = ti.max(rho_i, self.density_0)
                self.container.particle_densities[p_i] = rho_i
                self.container.particle_pressures[p_i] = self.stiffness * (ti.pow(rho_i / self.density_0, self.gamma) - 1.0)

    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_densities[p_i] = self.container.particle_reference_volumes[p_i] * self.cubic_kernel(0.0)
                density_i = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_task, density_i)
                self.container.particle_densities[p_i] += density_i
                self.container.particle_densities[p_i] *= self.density_0

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        # Fluid neighbors and rigid neighbors are the same here.
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        R_mod = R.norm()
        ret += self.container.particle_reference_volumes[p_j] * self.cubic_kernel(R_mod)

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
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
    

    @ti.kernel
    def update_rigid_body(self):
        self.container.rigid_body_forces.fill(0.0)
        # TODO:deal with torque
        
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_is_dynamic[p_i]:
                    object_id = self.container.particle_object_ids[p_i]
                    self.container.rigid_body_forces[object_id] += self.container.particle_densities[p_i] * self.container.V0 * self.container.particle_accelerations[p_i]
        
        for obj_i in range(self.container.rigid_body_num[None]):
            if self.container.rigid_body_is_dynamic[obj_i]:
                delta_v = self.dt[None] * self.container.rigid_body_forces[obj_i] / self.container.rigid_body_masses[obj_i]
                self.container.rigid_body_velocities[obj_i] += delta_v
                delta_x = self.dt[None] * self.container.rigid_body_velocities[obj_i]
                self.container.rigid_body_centers_of_mass[obj_i] += delta_x

        # for obj_i in range(self.container.rigid_body_num[None]):
        #     if self.container.rigid_body_is_dynamic[obj_i]:
        #         self.container.rigid_body_velocities[obj_i] += self.dt[None] * self.g
        #         self.container.rigid_body_centers_of_mass[obj_i] += self.dt[None] * self.container.rigid_body_velocities[obj_i]

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_is_dynamic[p_i]:
                    object_id = self.container.particle_object_ids[p_i]
                    self.container.particle_velocities[p_i] = self.container.rigid_body_velocities[object_id]
                    self.container.particle_positions[p_i] = self.container.rigid_body_centers_of_mass[object_id] + self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]

        # for p_i in range(self.container.particle_num[None]):
        #     if self.container.particle_materials[p_i] == self.container.material_rigid:
        #         if self.container.particle_is_dynamic[p_i]:
        #             self.container.particle_velocities[p_i] += self.g * self.dt[None]
        #             self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]


    def step(self):
        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.compute_pressure()
     
        self.compute_pressure_acceleration()
        self.update_velocities()
        self.update_position()
        self.update_rigid_body()
        self.enforce_boundary_3D(self.container.material_fluid)