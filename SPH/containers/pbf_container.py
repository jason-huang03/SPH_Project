import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from ..utils import SimConfig
from .base_container import BaseContainer


@ti.data_oriented
class PBFContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        self.particle_old_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_num[None])
        self.particle_pbf_lambdas = ti.field(dtype=ti.f32, shape=self.particle_num[None])