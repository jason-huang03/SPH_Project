import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from ..utils import SimConfig
from .base_container import BaseContainer

@ti.data_oriented
class IISPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        self.density_error = ti.field(dtype=float, shape=())

        self.iisph_aii = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.dii = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.pressure_lap = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_pressure_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)

        self.dij_pj = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.sum_i = ti.field(dtype=float, shape=self.particle_max_num)

        self.temp = ti.field(dtype=int, shape=())
        self.temp1 = ti.field(dtype=int, shape=())
        self.temp2 = ti.field(dtype=int, shape=())

