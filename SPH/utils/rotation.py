import taichi as ti


@ti.func
def calculate_skew(v: ti.types.vector(3, float)):
    return ti.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])