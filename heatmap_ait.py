
import pickle as pkl
import numpy as np
import torch
import plotly.io as pio
pio.renderers.default = 'browser'



from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer



npz_path = r"C:\Users\jcame\Downloads\models_smplx_v1_1\models\smpl\SMPL_MALE.npz"
pkl_path = r"C:\Users\jcame\Downloads\models_smplx_v1_1\models\smpl\SMPL_MALE.pkl"

with open(pkl_path, "wb") as f:
    data = np.load(npz_path, allow_pickle=True)  
    pkl.dump(dict(data), f)


C.smplx_models = r"C:\Users\jcame\Downloads\SMPL_python_v.1.0.0"
if __name__ == "__main__":
    # Load DIP-IMU data (pkl file extracted from DIP IMU and Others .zip)
    with open(r"C:\Users\jcame\Downloads\DIPIMUandOthers\DIP_IMU_and_Others\DIP_IMU\DIP_IMU\s_02\02.pkl", "rb") as f:
        data = pkl.load(f, encoding="latin1")

    all_sensors = True  # True = show all 17 IMUs, False = show DIP 6-IMU setup

    oris = data["imu_ori"]
    poses = data["gt"]
    gender = "male"

    poses = poses[::2]  # Downsample to 30Hz
    oris = oris[::2]

    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    
    
   #print("poses[:, 3:].shape:", poses[:, 3:].shape)
   #print("poses[:, :3].shape:", poses[:, :3].shape)

    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    print("Pose shape:", poses.shape)  # (num_frames, ?)


    # Run SMPL-X forward pass
    _, joints = smpl_layer(
    poses_body=torch.from_numpy(poses[:, 3:]).float().to(C.device),  # shape (2092, 69)
    poses_root=torch.from_numpy(poses[:, :3]).float().to(C.device),  # shape (2092, 3)
    betas=betas,  # shape (2092, 10)
)
    joints_np = joints.cpu().numpy()


    sensor_placement = [
        "head", "sternum", "pelvis", "lshoulder", "rshoulder",
        "lupperarm", "rupperarm", "llowerarm", "rlowerarm",
        "lupperleg", "rupperleg", "llowerleg", "rlowerleg",
        "lhand", "rhand", "lfoot", "rfoot",
    ]

    joint_idxs = [15, 12, 0, 13, 14, 16, 17, 20, 21, 1, 2, 4, 5, 22, 23, 10, 11]

    sensor_sub_idxs = [7, 8, 11, 12, 0, 2] if not all_sensors else list(range(len(joint_idxs)))
    rbs = RigidBodies(joints[:, joint_idxs][:, sensor_sub_idxs].cpu().numpy(), oris[:, sensor_sub_idxs])

    smpl_seq = SMPLSequence(
        poses_body=poses[:, 3:],
        smpl_layer=smpl_layer,
        poses_root=poses[:, :3]
    )
    

    smpl_seq.mesh_seq.color = smpl_seq.mesh_seq.color[:3] + (0.5,)
    

   

    v = Viewer()
    v.playback_fps = 30.0
    v.scene.add(smpl_seq, rbs)
    
import numpy as np
import plotly.graph_objects as go

# === Frame sampling settings ===
step = 600
frames_to_plot = list(range(step, len(joints_np), step))  # skip first since no previous frame

for frame in frames_to_plot:
    # Compute velocity (magnitude of position change)
    joint_velocities = np.linalg.norm(joints_np[frame] - joints_np[frame - 1], axis=1)

    # Normalize for coloring (optional)
    norm_velocities = (joint_velocities - joint_velocities.min()) / (joint_velocities.ptp() + 1e-6)

    joint_positions = joints_np[frame]

    # Plot with color mapped to joint velocity
    fig = go.Figure(data=[
        go.Scatter3d(
            x=joint_positions[:, 0],
            y=joint_positions[:, 1],
            z=joint_positions[:, 2],
            mode='markers+text',
            marker=dict(size=6, color=norm_velocities, colorscale='Viridis', colorbar=dict(title='Velocity')),
            text=[f'J{i}' for i in range(joint_positions.shape[0])],
            textposition='top center'
        )
    ])

    fig.update_layout(
        title=f"Joint Activity Heatmap - Frame {frame}",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()
import numpy as np
from collections import Counter
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres
from matplotlib import cm
import matplotlib



# === Use joints_np from actual SMPL sequence ===
# (already defined in your code above this block)
assert joints_np.ndim == 3  # Shape: (num_frames, num_joints, 3)

# === Voxel grid settings ===
voxel_size = 0.1  # meters (10cm cube)

# === Flatten all joint positions across all frames ===
all_points = joints_np.reshape(-1, 3)

# === Find voxel occupancy ===
min_bound = np.min(all_points, axis=0)
voxel_indices = np.floor((all_points - min_bound) / voxel_size).astype(int)
voxel_keys = [tuple(idx) for idx in voxel_indices]
voxel_counts = Counter(voxel_keys)

# === Normalize by number of frames ===
normalized = {k: v / joints_np.shape[0] for k, v in voxel_counts.items()}

# === Convert to 3D positions and colors ===
positions = []
colors = []

# Use matplotlib colormap for heatmap coloring
cmap = matplotlib.colormaps['viridis']

for (i, j, k), freq in normalized.items():
    center = min_bound + np.array([i, j, k]) * voxel_size + voxel_size / 2
    positions.append(center)

    # Get RGBA from colormap
    rgba = cmap(freq)
    colors.append([rgba[0], rgba[1], rgba[2], 1.0])  # Use full alpha

positions = np.array(positions)
colors = np.array(colors)

# === Create voxel heatmap as spheres ===
spheres = Spheres(
    positions=positions,
    radius=0.045,   # Slightly less than voxel size
    color=colors,
)

# === Add to viewer (add this after your existing viewer launch) ===
v.scene.add(spheres)
v.run()




