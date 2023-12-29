import bpy
import sys
import os

# Set rendering device to OPTIX
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
bpy.context.preferences.addons['cycles'].preferences.get_devices()

# set denoiser
denoiser = 'OPTIX'  # Replace 'OPTIX' with your desired denoiser
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.denoiser = denoiser
bpy.context.scene.cycles.denoising_optix_input_passes = 'RGB_ALBEDO_NORMAL'

rendering_device_type = sys.argv[-4]  # Assumes the rendering device type is the fourth last argument
gpu_id = int(sys.argv[-3])  # Assumes the optix id is the third last argument
device_id_count = 0

for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == rendering_device_type:
        if device_id_count == gpu_id:
            device.use = True
            print(f"using device: {device.name} id: {gpu_id} type: {device.type}")
        else:
            device.use = False
        device_id_count += 1
    else:
        device.use = False



# Get the .obj file path from command line arguments
frame_dir = sys.argv[-2] # Assumes the frame directory is the second last argument
output_image_path = sys.argv[-1]  # Assumes the output image path is the last argument


for file in os.listdir(frame_dir):
    if file.endswith(".obj"):
        obj_path = os.path.join(frame_dir, file)
        target_obj_name = file.split(".")[0]
        # Import the .obj file
        bpy.ops.import_scene.obj(filepath=obj_path)
        new_obj = bpy.context.selected_objects[0]
        new_obj.name = target_obj_name + "_new"
        # Check if the target object exists
        if target_obj_name in bpy.data.objects:
            target_obj = bpy.data.objects[target_obj_name]

            # Transfer materials from the target object to the new object
            new_obj.data.materials.clear()
            for mat in target_obj.data.materials:
                new_obj.data.materials.append(mat)

            # Delete the original object
            bpy.data.objects.remove(target_obj, do_unlink=True)

# Render the scene to the specified file
bpy.context.scene.render.filepath = output_image_path
bpy.ops.render.render(write_still=True)
