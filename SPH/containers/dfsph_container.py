import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from ..utils import SimConfig
from .base_container import BaseContainer

@ti.data_oriented
class DFSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        # additional dfsph related property
        self.particle_dfsph_alphas = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa_v = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_derivatives = ti.field(dtype=float, shape=self.particle_max_num)
