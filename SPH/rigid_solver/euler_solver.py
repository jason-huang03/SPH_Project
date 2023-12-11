import taichi as ti
import numpy as np
from ..containers import BaseContainer

@ti.data_oriented
class EulerRigidSolver():
    def __init__(self, container: BaseContainer):
        self.container = container