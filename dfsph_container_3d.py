import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig

@ti.data_oriented
class DFSPHContainer3D:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)

        # Material
        self.material_rigid = 0
        self.material_fluid = 1

        self.dx = 0.01  # particle radius
        self.dx = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.dx
        self.dh = self.dx * 4.0  # support radius
        self.V0 = 0.8 * self.particle_diameter ** self.dim

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.dh
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        rigid_body_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num
        


        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_body_particle_num += voxelized_points_np.shape[0]
        

        
        self.fluid_particle_num = fluid_particle_num
        self.rigid_body_particle_num = rigid_body_particle_num
        self.particle_max_num = fluid_particle_num + rigid_body_particle_num # TODO: make it able to add particles

        print(f"Fluid particle num: {self.fluid_particle_num}, Rigid body particle num: {self.rigid_body_particle_num}")

        #========== Allocate memory ==========#
        # Particle num of each grid
        self.grid_num_particles = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_num_particles_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_num_particles.shape[0])

        # Particle related properties
        self.particle_object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_positions = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_velocities = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_reference_volumes = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_masses = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_pressures = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_materials = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_colors = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.particle_is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)

        self.particle_dfsph_alphas = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa_v = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_derivatives = ti.field(dtype=float, shape=self.particle_max_num)

        # Buffer for sort
        self.particle_object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_positions_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_velocities_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_volumes_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_masses_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_materials_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_colors_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)


        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)


    ###### Add particles ######
        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, 
                          color=color,
                          material=1) # 1 indicates fluid

        # Rigid body
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)

            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32), # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color
    

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.particle_object_id[p] = obj_id
        self.particle_positions[p] = x
        self.x_0[p] = x
        self.particle_velocities[p] = v
        self.particle_densities[p] = density
        self.particle_reference_volumes[p] = self.V0
        self.particle_masses[p] = self.V0 * density
        self.particle_pressures[p] = pressure
        self.particle_materials[p] = material
        self.particle_is_dynamic[p] = is_dynamic
        self.particle_colors[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num


    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)


    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    

    @ti.func
    def is_static_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and (not self.particle_is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and self.particle_is_dynamic[p]
    

    @ti.kernel
    def init_grid(self):
        self.grid_num_particles.fill(0)
        for I in ti.grouped(self.particle_positions):
            grid_index = self.get_flatten_grid_index(self.particle_positions[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_num_particles[grid_index], 1)
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles_temp[I] = self.grid_num_particles[I]
    
    @ti.kernel
    def reorder_particles(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_num_particles[self.grid_ids[I]-1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_num_particles_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for I in ti.grouped(self.grid_ids):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.particle_object_id_buffer[new_index] = self.particle_object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.particle_positions_buffer[new_index] = self.particle_positions[I]
            self.particle_velocities_buffer[new_index] = self.particle_velocities[I]
            self.particle_volumes_buffer[new_index] = self.particle_reference_volumes[I]
            self.particle_masses_buffer[new_index] = self.particle_masses[I]
            self.particle_densities_buffer[new_index] = self.particle_densities[I]
            self.particle_materials_buffer[new_index] = self.particle_materials[I]
            self.particle_colors_buffer[new_index] = self.particle_colors[I]
            self.is_dynamic_buffer[new_index] = self.particle_is_dynamic[I]

        for I in ti.grouped(self.particle_positions):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.particle_object_id[I] = self.particle_object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.particle_positions[I] = self.particle_positions_buffer[I]
            self.particle_velocities[I] = self.particle_velocities_buffer[I]
            self.particle_reference_volumes[I] = self.particle_volumes_buffer[I]
            self.particle_masses[I] = self.particle_masses_buffer[I]
            self.particle_densities[I] = self.particle_densities_buffer[I]
            self.particle_materials[I] = self.particle_materials_buffer[I]
            self.particle_colors[I] = self.particle_colors_buffer[I]
            self.particle_is_dynamic[I] = self.is_dynamic_buffer[I]


    def prepare_neighborhood_search(self):
        self.init_grid()
        self.prefix_sum_executor.run(self.grid_num_particles)
        self.reorder_particles()
    

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.particle_positions[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            start_idx = 0
            end_idx = self.grid_num_particles[grid_index]
            if grid_index - 1 >= 0:
                start_idx = self.grid_num_particles[grid_index-1]
            for p_j in range(start_idx, end_idx):
                if p_i != p_j and (self.particle_positions[p_i] - self.particle_positions[p_j]).norm() < self.dh:
                    task(p_i, p_j, ret)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.particle_object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.particle_positions[i]
                self.color_vis_buffer[i] = self.particle_colors[i] / 255.0

    def dump(self, obj_id):
        np_object_id = self.particle_object_id.to_numpy()
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.particle_positions.to_numpy()[mask]
        np_v = self.particle_velocities.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        print('particle num ', num_new_particles)

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)