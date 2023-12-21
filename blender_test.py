# run "blender -b --python blender_test.py" to test available devices

import bpy

bpy.context.preferences.addons['cycles'].preferences.get_devices()
print("Cycles Devices:")
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    print(f"  {device.name}: {device.type}")
