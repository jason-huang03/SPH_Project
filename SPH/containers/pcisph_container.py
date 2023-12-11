import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from ..utils import SimConfig
from .base_container import BaseContainer


@ti.data_oriented
class PCISPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        # PCISPH related property
        self.particle_pcisph_k = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_pressure_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_change = ti.field(dtype=float, shape=self.particle_max_num)

        self.rigid_body_pressure_forces = ti.Vector.field(self.dim, dtype=float, shape=10)
        self.rigid_body_pressure_torques = ti.Vector.field(self.dim, dtype=float, shape=10)


