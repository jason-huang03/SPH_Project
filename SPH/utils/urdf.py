def create_urdf(mesh_path, mass, scale, urdf_path):
    """
    Create a basic URDF file for a mesh object.

    Parameters:
    mesh_path (str): Path to the mesh file.
    mass (float): Mass of the object.
    scale (tuple): Scale of the object (x, y, z).
    urdf_path (str): Path where the URDF file will be saved.
    """
    urdf_content = f"""
<robot name="custom_mesh_robot">
    <link name="baseLink">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
        </collision>
    </link>
</robot>
    """

    with open(urdf_path, 'w') as file:
        file.write(urdf_content.strip())
