# implementation of paper "Weakly compressible SPH for free surface flows"
import taichi as ti
import numpy as np
from ..containers import WCSPHContainer
from .base_solver import BaseSolver
@ti.data_oriented
class WCSPHSolver(BaseSolver):
    def __init__(self, container: WCSPHContainer):
        super().__init__(container)

        # wcsph related parameters
        self.gamma = 7.0
        self.stiffness = 50000.0


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
            if self.container.particle_is_dynamic[p_i]:
                self.container.particle_accelerations[p_i] = ti.Vector([0.0 for _ in range(self.container.dim)])
                if self.container.particle_materials[p_i] == self.container.material_fluid:
                    ret_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                    self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, ret_i)
                    self.container.particle_accelerations[p_i] = ret_i


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
                # TODO: there seems to be a problem here. 
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

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_is_dynamic[p_i]:
                    object_id = self.container.particle_object_ids[p_i]
                    self.container.particle_velocities[p_i] = self.container.rigid_body_velocities[object_id]
                    self.container.particle_positions[p_i] = self.container.rigid_body_centers_of_mass[object_id] + self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]

    def step(self):
        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocities()

        self.compute_pressure()
        self.compute_pressure_acceleration()
        self.update_fluid_velocities()
        
        self.update_fluid_position()
        self.update_rigid_body()
        self.enforce_boundary_3D(self.container.material_fluid)


    def prepare(self):
        self.compute_rigid_particle_volume()
