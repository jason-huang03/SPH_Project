# implementation of paper "Divergence-Free Smoothed Particle Hydrodynamics"
# fluid rigid interaction force implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
import taichi as ti
import numpy as np
from ..containers import DFSPHContainer
from .base_solver import BaseSolver
from .DFSPH import DFSPHSolver

@ti.data_oriented
class DFSPHSolverImplicitViscosity(DFSPHSolver):
    def __init__(self, container: DFSPHContainer):
        super().__init__(container)


    def compute_non_pressure_acceleration(self):
        self.compute_gravity_acceleration()
        self.compute_surface_tension_acceleration()

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

        self.implicit_viscosity_solve()

