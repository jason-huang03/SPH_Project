# implementation of paper "Weakly compressible SPH for free surface flows"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
from ..containers import WCSPHContainer
from ..rigid_solver import PyBulletSolver
from .base_solver import BaseSolver
@ti.data_oriented
class WCSPHSolver(BaseSolver):
    def __init__(self, container: WCSPHContainer):
        super().__init__(container)

        # wcsph related parameters
        self.gamma = 7.0
        self.stiffness = 50000.0

        self.rigid_solver = PyBulletSolver(container, gravity=self.g,  dt=self.dt[None])


    @ti.kernel
    def compute_pressure_acceleration(self):
        self.container.particle_accelerations.fill(0.0)
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
                * self.kernel_gradient(R)
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # use fluid particle pressure, density as rigid particle pressure, density
            den_j = self.container.particle_densities[p_i]
            acc = (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_i] / (den_j * den_j)) * self.kernel_gradient(R)
            )
            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j = - acc * self.container.particle_rest_volumes[p_j] * self.density_0
                torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j

    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                rho_i = self.container.particle_densities[p_i]
                rho_i = ti.max(rho_i, self.density_0)
                self.container.particle_densities[p_i] = rho_i
                self.container.particle_pressures[p_i] = self.stiffness * (ti.pow(rho_i / self.density_0, self.gamma) - 1.0)

 
    def step(self):
        self.init_rigid_body_force_and_torque()
        self.container.prepare_neighborhood_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()

        self.compute_pressure()
        self.compute_pressure_acceleration()
        self.update_fluid_velocity()
        self.update_fluid_position()
        
        self.rigid_solver.step()
        self.renew_rigid_particle_state()
    
        self.enforce_boundary_3D(self.container.material_fluid)


    def prepare(self):
        self.compute_rigid_particle_volume()
