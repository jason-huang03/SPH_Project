import bpy
import os
from math import pi
background_color = (0.5, 0.9, 0.5)
def setup_scene():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'  # Or 'CPU' depending on your preference

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    r = 2 * pi / 360 * 70
    p = 2 * pi / 360 * -1.89
    y = 2 * pi / 360 * 51

    bpy.ops.object.camera_add(location=(7.6, -1.5, 4.2), rotation=(r, p, y))
    bpy.context.scene.camera = bpy.context.object

    bpy.ops.object.light_add(type='SUN', location=(5.6, 3.6, 4.0))
    bpy.data.objects['Sun'].data.energy = 5

    # Delete default cube (if present)
    if "Cube" in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

def create_particle_material():
    mat = bpy.data.materials.new(name="ParticleMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Roughness'].default_value = 0.2
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Base Color'].default_value = (0.9, 0.4, 1.0, 1)
    return mat

# Set the path to your PLY file and the output directory
ply_file_path = '/home/jason/SPH_Project/high_fluid_dfsph_output/particle_object_0_000000.ply'
output_directory = 'rendered_images/test.png'

# Setup the scene
setup_scene()
bpy.context.scene.world.use_nodes = False  # Disable nodes to set a solid color
bpy.context.scene.world.color = background_color
# Create particle material
particle_material = create_particle_material()

# Import the PLY file
bpy.ops.import_mesh.ply(filepath=ply_file_path)
obj = bpy.context.selected_objects[0]

# Apply the particle material
obj.data.materials.append(particle_material)

# Scale the object (adjust as needed)
# obj.scale = (0.02, 0.02, 0.02)  # Scale down the particles

# Set render output path
bpy.context.scene.render.filepath = os.path.join(output_directory, "rendered_image.png")

# Render the frame
bpy.ops.render.render(write_still=True)

# Remove the object to clear memory
bpy.data.objects.remove(obj, do_unlink=True)
