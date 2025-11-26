import argparse
import base64
import io
import json
import mmcv
import numpy as np
import requests
import torch
from pyquaternion import Quaternion

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

def load_image_as_base64_png_tensor(img_path):
    """
    NeuroNCAP’s server expects images serialized with torch.save (NOT raw PNG).
    So we:
    1. load image as RGB
    2. convert to numpy
    3. dump to torch.save into a BytesIO buffer
    4. base64 encode that buffer
    """
    img = mmcv.imread(img_path)[..., ::-1]  # BGR → RGB
    tensor = torch.from_numpy(img)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    png_bytes = buffer.getvalue()
    return base64.b64encode(png_bytes).decode("utf-8")


def prepare_request_from_info(info, data_root):
    # --------------------------------------------------------
    # 1. Load 6 camera images
    # --------------------------------------------------------
    images_b64 = {}
    for cam in NUSCENES_CAM_ORDER:
        cam_info = info["cams"][cam]
        img_path = cam_info["data_path"]
        if not img_path.startswith("/"):
            img_path = f"{data_root}/{img_path}"
        images_b64[cam] = load_image_as_base64_png_tensor(img_path)

    # --------------------------------------------------------
    # 2. ego2world
    # --------------------------------------------------------
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3]  = info["ego2global_translation"]

    # --------------------------------------------------------
    # 3. lidar2ego
    # --------------------------------------------------------
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3]  = info["lidar2ego_translation"]

    # --------------------------------------------------------
    # 4. camera2ego and camera intrinsics
    # --------------------------------------------------------
    camera2image = {}
    camera2ego = {}

    for cam in NUSCENES_CAM_ORDER:
        cam_info = info["cams"][cam]

        # camera intrinsic
        K = np.array(cam_info["cam_intrinsic"], dtype=np.float32)

        # sensor2ego
        T = np.eye(4)
        T[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
        T[:3, 3]  = cam_info["sensor2ego_translation"]

        camera2image[cam] = K.tolist()
        camera2ego[cam] = T.tolist()

    # --------------------------------------------------------
    # 5. CAN bus = ego_status for DiffusionDrive
    # --------------------------------------------------------
    ego_status = info["ego_status"]  # length 10 vector

    # --------------------------------------------------------
    # 6. Build final request dict
    # --------------------------------------------------------
    request_payload = {
        "images": images_b64,
        "ego2world": ego2global.tolist(),
        "canbus": ego_status,
        "timestamp": int(info["timestamp"]),  # microseconds
        "command": 2,  # straight (or None)
        "calibration": {
            "camera2image": camera2image,
            "camera2ego": camera2ego,
            "lidar2ego": lidar2ego.tolist(),
        }
    }

    return request_payload


def main(args):
    print("Loading NuScenes infos...")
    data = mmcv.load(args.ann_file)
    infos = sorted(data["infos"], key=lambda e: (e["scene_token"], e["timestamp"]))

    info = infos[args.frame]

    print("Preparing request payload...")
    payload = prepare_request_from_info(info, args.data_root)

    url = f"http://{args.server_host}:{args.server_port}/infer"
    print(f"POST {url}")

    resp = requests.post(url, json=payload)

    print("\n===== SERVER RESPONSE =====")
    if resp.status_code != 200:
        print("ERROR:", resp.status_code)
        print(resp.text)
        return

    result = resp.json()
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", required=True, help="nusc infos .pkl")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", default="9000")
    args = parser.parse_args()
    main(args)
