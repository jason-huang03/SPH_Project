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
