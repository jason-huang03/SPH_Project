import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from ..utils import SimConfig
from .base_container import BaseContainer


@ti.data_oriented
class WCSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        