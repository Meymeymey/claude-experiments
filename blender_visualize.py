"""
Blender visualization script using bpy.
Creates 3D visualization of the hydrogen/electricity network with flow animation.

Usage:
    1. Open Blender
    2. Go to Scripting workspace
    3. Open this file and run it
    OR
    Run from command line: blender --python blender_visualize.py
"""

import json
import math
import os

# bpy is only available when running inside Blender
try:
    import bpy
    import bmesh
    from mathutils import Vector
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False
    print("Warning: bpy not available. Run this script inside Blender.")


# Configuration
NODE_COLORS = {
    'producer': (0.2, 0.8, 0.2, 1.0),    # Green for producers
    'converter': (0.8, 0.8, 0.2, 1.0),   # Yellow for converters
    'consumer': (0.8, 0.2, 0.2, 1.0),    # Red for consumers
}

PIPE_COLORS = {
    'electricity': (1.0, 0.8, 0.0, 1.0),  # Yellow/Gold for electricity
    'hydrogen': (0.2, 0.6, 1.0, 1.0),     # Blue for hydrogen
}

FLOW_COLORS = {
    'electricity': (1.0, 0.9, 0.3, 1.0),  # Bright yellow for electricity flow
    'hydrogen': (0.4, 0.8, 1.0, 1.0),     # Bright blue for hydrogen flow
}

NODE_SCALE = 0.6
NODE_HEIGHT = 0.3  # Extrusion height for flat-top shapes
PIPE_RADIUS = 0.06
FLOW_PARTICLE_RADIUS = 0.08
ANIMATION_FRAMES = 120  # Total animation length
PARTICLES_PER_PIPE = 5  # Number of flow particles per pipe


def clear_scene():
    """Remove all objects from the current scene."""
    if not BPY_AVAILABLE:
        return

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def create_material(name: str, color: tuple, emission: float = 0.0) -> 'bpy.types.Material':
    """Create a material with the given color and optional emission."""
    if not BPY_AVAILABLE:
        return None

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.5
        if emission > 0:
            bsdf.inputs['Emission Color'].default_value = color
            bsdf.inputs['Emission Strength'].default_value = emission

    return mat


def create_extruded_triangle(name: str, position: tuple, scale: float, height: float) -> 'bpy.types.Object':
    """Create an extruded triangle (triangular prism) with flat top and bottom."""
    if not BPY_AVAILABLE:
        return None

    # Create mesh and bmesh
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()

    # Define triangle vertices (equilateral triangle on XY plane)
    angle_offset = math.pi / 2  # Point upward
    verts_bottom = []
    verts_top = []

    for i in range(3):
        angle = angle_offset + (2 * math.pi * i / 3)
        x = scale * math.cos(angle)
        y = scale * math.sin(angle)
        verts_bottom.append(bm.verts.new((x, y, -height / 2)))
        verts_top.append(bm.verts.new((x, y, height / 2)))

    bm.verts.ensure_lookup_table()

    # Create faces
    # Bottom face
    bm.faces.new(verts_bottom[::-1])  # Reversed for correct normal
    # Top face
    bm.faces.new(verts_top)
    # Side faces
    for i in range(3):
        next_i = (i + 1) % 3
        bm.faces.new([verts_bottom[i], verts_bottom[next_i], verts_top[next_i], verts_top[i]])

    bm.to_mesh(mesh)
    bm.free()

    # Create object
    obj = bpy.data.objects.new(name, mesh)
    obj.location = position
    bpy.context.collection.objects.link(obj)

    return obj


def create_extruded_box(name: str, position: tuple, scale: float, height: float) -> 'bpy.types.Object':
    """Create an extruded box (rectangular prism) with flat top and bottom."""
    if not BPY_AVAILABLE:
        return None

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=position
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (scale, scale, height / 2)

    # Apply scale
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)

    return obj


def create_extruded_circle(name: str, position: tuple, scale: float, height: float) -> 'bpy.types.Object':
    """Create an extruded circle (cylinder) with flat top and bottom."""
    if not BPY_AVAILABLE:
        return None

    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale,
        depth=height,
        location=position,
        vertices=32  # Smooth circle
    )
    obj = bpy.context.active_object
    obj.name = name

    return obj


def create_node_mesh(name: str, position: tuple, node_type: str, carrier: str) -> 'bpy.types.Object':
    """Create a 3D node representation with extruded flat shapes."""
    if not BPY_AVAILABLE:
        return None

    # Create different shapes based on node type
    if node_type == 'producer':
        # Extruded triangle for producers
        obj = create_extruded_triangle(name, position, NODE_SCALE, NODE_HEIGHT)
    elif node_type == 'converter':
        # Extruded box for converters
        obj = create_extruded_box(name, position, NODE_SCALE, NODE_HEIGHT)
    else:  # consumer
        # Extruded circle for consumers
        obj = create_extruded_circle(name, position, NODE_SCALE * 0.8, NODE_HEIGHT)

    if obj is None:
        return None

    # Apply material
    mat = create_material(f"{name}_material", NODE_COLORS.get(node_type, (0.5, 0.5, 0.5, 1.0)))
    obj.data.materials.append(mat)

    # Add text label
    create_label(name, position)

    return obj


def create_label(text: str, position: tuple) -> 'bpy.types.Object':
    """Create a floating text label above a node."""
    if not BPY_AVAILABLE:
        return None

    # Create text object - positioned above node on Z axis
    bpy.ops.object.text_add(
        location=(position[0], position[1], position[2] + NODE_HEIGHT / 2 + 0.4)
    )
    text_obj = bpy.context.active_object
    text_obj.data.body = text
    text_obj.data.size = 0.35
    text_obj.data.align_x = 'CENTER'
    text_obj.name = f"{text}_label"

    return text_obj


def create_pipe(name: str, start_pos: tuple, end_pos: tuple, carrier: str) -> 'bpy.types.Object':
    """Create a pipe/cable between two points."""
    if not BPY_AVAILABLE:
        return None

    # Calculate pipe properties
    start = Vector(start_pos)
    end = Vector(end_pos)
    direction = end - start
    length = direction.length
    midpoint = (start + end) / 2

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=PIPE_RADIUS,
        depth=length,
        location=midpoint
    )

    obj = bpy.context.active_object
    obj.name = name

    # Rotate cylinder to align with direction
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    # Apply material (slightly transparent to see flow)
    mat = create_material(f"{name}_material", PIPE_COLORS.get(carrier, (0.5, 0.5, 0.5, 1.0)))
    obj.data.materials.append(mat)

    return obj


def create_flow_particle(name: str, carrier: str) -> 'bpy.types.Object':
    """Create a single flow particle (small glowing sphere)."""
    if not BPY_AVAILABLE:
        return None

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=FLOW_PARTICLE_RADIUS,
        segments=12,
        ring_count=8,
        location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name

    # Create glowing material
    color = FLOW_COLORS.get(carrier, (1.0, 1.0, 1.0, 1.0))
    mat = create_material(f"{name}_material", color, emission=5.0)
    obj.data.materials.append(mat)

    return obj


def animate_flow_particle(particle: 'bpy.types.Object', start_pos: Vector, end_pos: Vector,
                          start_frame: int, end_frame: int):
    """Animate a flow particle moving from start to end position."""
    if not BPY_AVAILABLE or particle is None:
        return

    # Keyframe at start position
    particle.location = start_pos
    particle.keyframe_insert(data_path="location", frame=start_frame)

    # Keyframe at end position
    particle.location = end_pos
    particle.keyframe_insert(data_path="location", frame=end_frame)

    # Make animation linear
    if particle.animation_data and particle.animation_data.action:
        for fcurve in particle.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def create_flow_animation(edges: list, node_positions: dict):
    """Create animated flow particles along all pipes."""
    if not BPY_AVAILABLE:
        return

    # Set up animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = ANIMATION_FRAMES

    particle_index = 0

    for edge in edges:
        source = edge['source']
        target = edge['target']
        carrier = edge['carrier']

        start_pos = Vector(node_positions[source])
        end_pos = Vector(node_positions[target])

        # Calculate direction and offset positions to start/end at node edges
        direction = (end_pos - start_pos).normalized()
        adjusted_start = start_pos + direction * NODE_SCALE
        adjusted_end = end_pos - direction * NODE_SCALE

        # Create multiple particles with staggered timing
        frames_per_particle = ANIMATION_FRAMES // PARTICLES_PER_PIPE
        travel_frames = ANIMATION_FRAMES // 2  # Time to travel the pipe

        for i in range(PARTICLES_PER_PIPE):
            particle = create_flow_particle(f"Flow_{source}_{target}_{i}", carrier)

            if particle:
                # Calculate staggered start time
                start_frame = (i * frames_per_particle) % ANIMATION_FRAMES + 1

                # Animate the particle
                # First cycle
                animate_flow_particle(
                    particle,
                    adjusted_start,
                    adjusted_end,
                    start_frame,
                    start_frame + travel_frames
                )

                # Loop the animation by adding more keyframes
                loop_start = start_frame + travel_frames + 1
                if loop_start <= ANIMATION_FRAMES:
                    # Hide briefly then restart
                    particle.location = adjusted_start
                    particle.keyframe_insert(data_path="location", frame=loop_start)

                particle_index += 1

    print(f"Created {particle_index} flow particles")


def setup_camera_and_lighting():
    """Set up camera and lighting for the scene (top-down view of XY plane)."""
    if not BPY_AVAILABLE:
        return

    # Add camera - looking down at XY plane from above
    bpy.ops.object.camera_add(
        location=(0, 0, 12),
        rotation=(0, 0, 0)
    )
    camera = bpy.context.active_object
    camera.name = 'Main_Camera'
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 10
    bpy.context.scene.camera = camera

    # Add sun light
    bpy.ops.object.light_add(
        type='SUN',
        location=(5, 5, 10)
    )
    sun = bpy.context.active_object
    sun.name = 'Sun'
    sun.data.energy = 3

    # Add ambient light
    bpy.ops.object.light_add(
        type='AREA',
        location=(0, 0, 8)
    )
    area = bpy.context.active_object
    area.name = 'Ambient'
    area.data.energy = 50
    area.data.size = 10


def create_legend():
    """Create a legend explaining the shapes (positioned for top-down XY view)."""
    if not BPY_AVAILABLE:
        return

    legend_x = 4
    legend_y = 2
    legend_scale = 0.25
    legend_height = 0.15

    # Producer legend - triangle
    obj = create_extruded_triangle('Legend_Producer', (legend_x, legend_y, 0), legend_scale, legend_height)
    if obj:
        mat = create_material('legend_producer_mat', NODE_COLORS['producer'])
        obj.data.materials.append(mat)

    # Converter legend - box
    obj = create_extruded_box('Legend_Converter', (legend_x, legend_y - 0.7, 0), legend_scale, legend_height)
    if obj:
        mat = create_material('legend_converter_mat', NODE_COLORS['converter'])
        obj.data.materials.append(mat)

    # Consumer legend - circle
    obj = create_extruded_circle('Legend_Consumer', (legend_x, legend_y - 1.4, 0), legend_scale * 0.8, legend_height)
    if obj:
        mat = create_material('legend_consumer_mat', NODE_COLORS['consumer'])
        obj.data.materials.append(mat)


def load_network_from_json(filepath: str) -> dict:
    """Load network data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_visualization_from_data(network_data: dict, animate: bool = True):
    """Build the complete visualization from network data."""
    if not BPY_AVAILABLE:
        print("Cannot build visualization: bpy not available")
        return

    # Clear existing scene
    clear_scene()

    # Create nodes
    node_positions = {}
    for node in network_data['nodes']:
        pos = tuple(node['position'])
        node_positions[node['name']] = pos
        create_node_mesh(
            name=node['name'],
            position=pos,
            node_type=node['node_type'],
            carrier=node['carrier']
        )

    # Create pipes
    for edge in network_data['edges']:
        start_pos = node_positions[edge['source']]
        end_pos = node_positions[edge['target']]

        # Offset start/end slightly to not overlap with nodes
        start_vec = Vector(start_pos)
        end_vec = Vector(end_pos)
        direction = (end_vec - start_vec).normalized()

        adjusted_start = start_vec + direction * NODE_SCALE
        adjusted_end = end_vec - direction * NODE_SCALE

        create_pipe(
            name=f"Pipe_{edge['source']}_{edge['target']}",
            start_pos=tuple(adjusted_start),
            end_pos=tuple(adjusted_end),
            carrier=edge['carrier']
        )

    # Create flow animation
    if animate:
        create_flow_animation(network_data['edges'], node_positions)

    # Setup scene
    setup_camera_and_lighting()
    create_legend()

    print("Visualization created successfully!")
    if animate:
        print(f"Animation: {ANIMATION_FRAMES} frames. Press Space to play.")


def build_default_visualization():
    """Build visualization with default network (no JSON file needed)."""
    # Default network data matching our system - flat on XY plane (Z=0)
    network_data = {
        'nodes': [
            {'name': 'Alpha', 'node_type': 'producer', 'carrier': 'electricity', 'position': [0.0, 2.0, 0.0]},
            {'name': 'Bravo', 'node_type': 'converter', 'carrier': 'both', 'position': [-2.0, 0.0, 0.0]},
            {'name': 'Charlie', 'node_type': 'consumer', 'carrier': 'hydrogen', 'position': [-2.0, -2.0, 0.0]},
            {'name': 'Delta', 'node_type': 'consumer', 'carrier': 'electricity', 'position': [2.0, 0.0, 0.0]},
        ],
        'edges': [
            {'source': 'Alpha', 'target': 'Bravo', 'carrier': 'electricity'},
            {'source': 'Alpha', 'target': 'Delta', 'carrier': 'electricity'},
            {'source': 'Bravo', 'target': 'Charlie', 'carrier': 'hydrogen'},
        ]
    }

    build_visualization_from_data(network_data, animate=True)


def save_blend_file(filepath: str = None):
    """Save the current scene as a .blend file."""
    if not BPY_AVAILABLE:
        return

    if filepath is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(script_dir, 'hydrogen_network.blend')

    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"Saved Blender file to: {filepath}")


# Main execution
if __name__ == "__main__":
    if BPY_AVAILABLE:
        # Check if network_data.json exists
        script_dir = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(script_dir, 'network_data.json')
        blend_path = os.path.join(script_dir, 'hydrogen_network.blend')

        if os.path.exists(json_path):
            print(f"Loading network from {json_path}")
            network_data = load_network_from_json(json_path)
            build_visualization_from_data(network_data, animate=True)
        else:
            print("No network_data.json found, using default network")
            build_default_visualization()

        # Save the .blend file
        save_blend_file(blend_path)

    else:
        print("\nThis script must be run inside Blender.")
        print("Usage:")
        print("  1. Open Blender")
        print("  2. Go to Scripting workspace")
        print("  3. Open this file and click 'Run Script'")
        print("\nOr run from command line:")
        print("  blender --python blender_visualize.py")
        print("\nThis will create 'hydrogen_network.blend' in the same folder.")
