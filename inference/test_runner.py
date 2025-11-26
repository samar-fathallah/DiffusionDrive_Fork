import argparse
import mmcv
import numpy as np
import torch
import os
import tempfile
from pyquaternion import Quaternion

from runner import (
    DiffusionDriveRunner,
    DiffusionDriveInferenceInput,
)

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]



# ----------------------------------------------------------
#  Utility: Save RGB numpy images to temporary PNG files
#  (because MMDet pipeline needs img_filename paths)
# ----------------------------------------------------------
def save_numpy_image_to_temp(img_np):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    mmcv.imwrite(img_np[..., ::-1], tmp.name)  # convert RGB→BGR for mmcv.imwrite
    tmp.close()
    return tmp.name


# ----------------------------------------------------------
#  Load a raw NuScenes info entry
#  BUT images are loaded manually and saved as temp files
#  Everything else (resize/normalize/etc) is done by pipeline
# ----------------------------------------------------------
def load_nuscenes_frame(info, data_root):
    img_paths = []
    cam_intrinsics = []
    lidar2img_list = []
    img_shapes = [] 
    # ------------------------------------------------------
    # 1. Load images from NuScenes paths (convert paths)
    # ------------------------------------------------------
    for cam in NUSCENES_CAM_ORDER:
        cam_info = info["cams"][cam]
        img_path = cam_info["data_path"]

        if not os.path.exists(img_path):
            img_path = img_path.replace("samples/", "v1.0-trainval/samples/")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = mmcv.imread(img_path)  # BGR
        img = img[..., ::-1]         # RGB

        temp_file = save_numpy_image_to_temp(img)
        img_paths.append(temp_file)
        h, w = img.shape[:2]
        img_shapes.append((h, w))  

    # ------------------------------------------------------
    # 2. Compute lidar2img exactly as DiffusionDrive expects
    # ------------------------------------------------------
    import copy
    for cam in NUSCENES_CAM_ORDER:
        cam_info = info["cams"][cam]

        # lidar2cam
        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T

        lidar2cam_rt = np.eye(4, dtype=np.float32)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        # intrinsics
        intrinsic = copy.deepcopy(cam_info["cam_intrinsic"]).astype(np.float32)
        cam_intrinsics.append(intrinsic)

        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = intrinsic

        lidar2img = viewpad @ lidar2cam_rt.T
        lidar2img_list.append(lidar2img)

    lidar2img = np.stack(lidar2img_list, axis=0)  # (6,4,4)
    cam_intrinsics = np.stack(cam_intrinsics, axis=0)

    # ------------------------------------------------------
    # 3. Compute lidar_pose = ego2global @ lidar2ego
    # ------------------------------------------------------
    lidar2ego_T = np.eye(4)
    lidar2ego_T[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego_T[:3, 3] = info["lidar2ego_translation"]

    ego2global_T = np.eye(4)
    ego2global_T[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global_T[:3, 3] = info["ego2global_translation"]

    lidar_pose = ego2global_T @ lidar2ego_T

    # ------------------------------------------------------
    # 4. Ego status + timestamp
    # ------------------------------------------------------
    ego_status = np.array(info["ego_status"], dtype=np.float32)
    timestamp = info["timestamp"] / 1e6

    return img_paths, lidar2img, lidar_pose, ego_status, timestamp, cam_intrinsics,img_shapes


# ----------------------------------------------------------
#  MAIN
# --------------------------------------------------------
def main(args):
    print("Loading config and runner…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = DiffusionDriveRunner(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print("Loading NuScenes infos…")
    data = mmcv.load(args.ann_file)
    infos = sorted(data["infos"], key=lambda e: (e["scene_token"], e["timestamp"]))

    print(f"Using frame {args.frame_idx}/{len(infos)}")
    info = infos[args.frame_idx]

    # ------------------------------------------------------
    #  Load raw frame into MMDet-style runner input
    # ------------------------------------------------------
    img_paths, lidar2img, lidar_pose, ego_status, timestamp, cam_intrinsics, img_shapes = \
        load_nuscenes_frame(info, args.data_root)

    dd_input = DiffusionDriveInferenceInput(
        img_paths=img_paths,          # IMPORTANT: these are file paths
        lidar_pose=lidar_pose,
        lidar2img=lidar2img,
        ego_status=ego_status,
        timestamp=timestamp,
        cam_intrinsic=cam_intrinsics,
    )
    # --- SANITY: print steering + velocity magnitude ---
    steer = float(dd_input.ego_status[9]) if dd_input.ego_status.size >= 10 else float('nan')
    vel   = dd_input.ego_status[6:9] if dd_input.ego_status.size >= 9 else np.zeros(3, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    print(f"[sanity] steer(deg)={steer:.3f}  vel(ego)={vel.tolist()}  speed={speed:.3f} m/s")
    if speed < 0.1:
        print("[sanity][WARN] speed is ~0 => planning may be nearly stationary")

    print("Running inference…")
    output = runner.forward_inference(dd_input)

    print("\n========= RESULTS =========")
    print("Trajectory (final_planning):")
    print(output.trajectory)
    # --- SANITY: cumulative displacement of the plan ---
    fp = output.trajectory  # shape (6, 2), per-step XY offsets in meters
    cum = fp.cumsum(axis=0)
    end = cum[-1]
    tot_len = float(np.linalg.norm(end))
    print(f"[sanity] endpoint(m)={end.tolist()}  total_length(m)={tot_len:.3f}")
    if tot_len < 0.1:
        print("[sanity][WARN] trajectory length < 0.1 m — likely due to ~0 speed prior")


    print("\nAux outputs:")
    aux = output.aux_outputs.to_json()
    for k, v in aux.items():
        if v is not None:
            print(f"  {k}: shape={np.array(v).shape}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()
    main(args)
