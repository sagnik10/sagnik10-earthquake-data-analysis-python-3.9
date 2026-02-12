import math
import pygltflib
from pygltflib import GLTF2, Scene, Node, Mesh, Material
from pygltflib import (
    Buffer, BufferView, Accessor, Animation, 
    AnimationSampler, AnimationChannel, AnimationChannelTarget
)
from pygltflib import ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER
from pygltflib import FLOAT, UNSIGNED_SHORT
from pygltflib import Primitive, Skin

import struct

# Helper to multiply two quaternions (x, y, z, w).
def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return (x, y, z, w)

# -----------------------------
# 1) Define geometry for a simple line
#    Make it a bit longer so the swing is more obvious.
# -----------------------------
positions = [
    0.0,  0.0,  0.0,  # pivot (origin)
    0.0, -2.0,  0.0   # longer pendulum tip
]
# Indices for a line segment
indices = [0, 1]

positions_bytes = b"".join([struct.pack("<f", p) for p in positions])
indices_bytes   = b"".join([struct.pack("<H", i) for i in indices])
combined_buffer = positions_bytes + indices_bytes

position_buffer_view = BufferView(
    buffer=0,
    byteOffset=0,
    byteLength=len(positions_bytes),
    target=ARRAY_BUFFER
)
index_buffer_view = BufferView(
    buffer=0,
    byteOffset=len(positions_bytes),
    byteLength=len(indices_bytes),
    target=ELEMENT_ARRAY_BUFFER
)

position_accessor = Accessor(
    bufferView=0,
    byteOffset=0,
    componentType=FLOAT,
    count=len(positions)//3,
    type="VEC3",
    min=[0.0, -2.0,  0.0],
    max=[0.0,  0.0,  0.0]
)
index_accessor = Accessor(
    bufferView=1,
    byteOffset=0,
    componentType=UNSIGNED_SHORT,
    count=len(indices),
    type="SCALAR",
    min=[0],
    max=[1]
)

mesh_primitive = Primitive(
    attributes={"POSITION": 0},
    indices=1,
    mode=pygltflib.LINES
)
mesh = Mesh(primitives=[mesh_primitive])

pendulum_node = Node(
    mesh=0,
    name="PendulumNode"
)

# -----------------------------
# 5) Build a more pronounced 3D motion animation.
# -----------------------------
period = 2.0
num_samples = 10

times = []
rotations = []

# Larger amplitude around Z, plus a small amplitude around X.
amplitude_degs_z = 45  # ±45 degrees around Z
amplitude_degs_x = 10  # ±10 degrees around X

omega = 2 * math.pi / period
amplitude_rads_z = math.radians(amplitude_degs_z)
amplitude_rads_x = math.radians(amplitude_degs_x)

for i in range(num_samples):
    t = i * (period / (num_samples - 1))
    # Angle around Z
    angle_z = amplitude_rads_z * math.cos(omega * t)
    # Angle around X
    angle_x = amplitude_rads_x * math.sin(omega * t)
    
    # Convert each to a quaternion:
    #   qZ for rotation around Z
    #   qX for rotation around X
    half_z = angle_z / 2
    half_x = angle_x / 2
    
    sin_z = math.sin(half_z)
    cos_z = math.cos(half_z)
    # rotation around Z = (0, 0, sin(θ/2), cos(θ/2))
    qZ = (0.0, 0.0, sin_z, cos_z)
    
    sin_x = math.sin(half_x)
    cos_x = math.cos(half_x)
    # rotation around X = (sin(θ/2), 0, 0, cos(θ/2))
    qX = (sin_x, 0.0, 0.0, cos_x)
    
    # Combine them: qTotal = qZ * qX
    qTotal = quaternion_multiply(qZ, qX)
    
    times.append(t)
    rotations.append(qTotal)

time_bytes = b"".join([struct.pack("<f", t) for t in times])
rotation_bytes = b"".join([struct.pack("<4f", *r) for r in rotations])

animation_time_buffer_view = BufferView(
    buffer=0,
    byteOffset=len(combined_buffer),
    byteLength=len(time_bytes),
    target=None
)
animation_rotation_buffer_view = BufferView(
    buffer=0,
    byteOffset=len(combined_buffer) + len(time_bytes),
    byteLength=len(rotation_bytes),
    target=None
)

time_accessor = Accessor(
    bufferView=2,
    byteOffset=0,
    componentType=FLOAT,
    count=len(times),
    type="SCALAR",
    min=[times[0]],
    max=[times[-1]]
)
rotation_accessor = Accessor(
    bufferView=3,
    byteOffset=0,
    componentType=FLOAT,
    count=len(rotations),
    type="VEC4"
)

animation_sampler = AnimationSampler(
    input=2,
    output=3,
    interpolation="LINEAR"
)
animation_channel = AnimationChannel(
    sampler=0,
    target=AnimationChannelTarget(
        node=0,
        path="rotation"
    )
)
animation = Animation(
    samplers=[animation_sampler],
    channels=[animation_channel],
    name="PendulumSwing"
)

gltf = GLTF2(
    scenes=[Scene(nodes=[0])],
    nodes=[pendulum_node],
    meshes=[mesh],
    animations=[animation],
    buffers=[Buffer(byteLength=len(combined_buffer) + len(time_bytes) + len(rotation_bytes))],
    bufferViews=[
        position_buffer_view, 
        index_buffer_view,
        animation_time_buffer_view,
        animation_rotation_buffer_view
    ],
    accessors=[
        position_accessor,
        index_accessor,
        time_accessor,
        rotation_accessor
    ]
)

# Store the combined binary data in the first buffer
gltf.buffers[0].uri = None
gltf._glb_data = combined_buffer + time_bytes + rotation_bytes

gltf.save("pendulum.glb")
print("pendulum.glb has been created.")

