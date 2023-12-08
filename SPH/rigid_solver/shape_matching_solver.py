import taichi as ti
import numpy as np
from ..containers import BaseContainer

class ShapeMatchingRigidSolver():
    def __init__(self, container: BaseContainer):
        self.container = container

