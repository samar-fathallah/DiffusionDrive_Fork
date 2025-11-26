import os
import io
import tempfile
from typing import Dict, List, Optional, Literal, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Base64Bytes, validator
from PIL import Image

# ---- Import DiffusionDrive runner + packer ----
from inference.runner import (
    DiffusionDriveRunner,
    DiffusionDriveInferenceInput,
    pack_outputs_for_neuroncap,
)

# ==========================
# Constants / config
# ==========================

CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CFG_PATH = os.environ.get("DD_CONFIG", "configs/diffusiondrive_config.py")
CKPT_PATH = os.environ.get("DD_CHECKPOINT", "ckpts/diffusiondrive_stage2.pth")

# ==========================
# Pydantic models
# ==========================

class Calibration(BaseModel):
    """Calibration data."""

    camera2image: Dict[str, List[List[float]]]
    """Camera intrinsics (3x3 or 4x4). Keys are camera names."""
    camera2ego: Dict[str, List[List[float]]]
    """Camera extrinsics (4x4). Keys are camera names."""
    lidar2ego: List[List[float]]
    """Lidar extrinsics (4x4)."""

    @validator("camera2image")
    def _check_cam2img(cls, v):
        for cam, mat in v.items():
            arr = np.array(mat, dtype=np.float32)
            if arr.shape not in [(3, 3), (4, 4)]:
                raise ValueError(f"camera2image[{cam}] must be 3x3 or 4x4, got {arr.shape}")
        return v

    @validator("camera2ego")
    def _check_cam2ego(cls, v):
        for cam, mat in v.items():
            arr = np.array(mat, dtype=np.float32)
            if arr.shape != (4, 4):
                raise ValueError(f"camera2ego[{cam}] must be 4x4, got {arr.shape}")
        return v

    @validator("lidar2ego")
    def _check_l2e(cls, v):
        arr = np.array(v, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"lidar2ego must be 4x4, got {arr.shape}")
        return v


class InferenceInputs(BaseModel):
    """Input data for inference (NeuroNCAP-spec)."""

    images: Dict[str, Base64Bytes]
    """Camera images encoded as Base64Bytes (torch-saved tensors). Keys = camera names."""
    ego2world: List[List[float]]
    """4x4 ego pose in world frame."""
    canbus: Any
    """CAN bus / ego-state vector (dict or list)."""
    timestamp: int
    """Timestamp in microseconds."""
    command: Optional[Literal[0, 1, 2]] = None
    """0: right, 1: left, 2: straight"""
    calibration: Calibration

    @validator("ego2world")
    def _check_e2w(cls, v):
        arr = np.array(v, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError("ego2world must be 4x4")
        return v


class InferenceAuxOutputs(BaseModel):
    objects_in_bev: Optional[List[List[float]]] = None  # N x [x, y, width, length, yaw]
    object_classes: Optional[List[str]] = None          # (N, )
    object_scores: Optional[List[float]] = None         # (N, )
    object_ids: Optional[List[int]] = None              # (N, )
    future_trajs: Optional[List[List[List[List[float]]]]] = None  # N x M x T x [x, y]


class InferenceOutputs(BaseModel):
    """Output / result from running the model."""
    trajectory: List[List[float]]
    """Predicted ego trajectory in the ego frame. List of (x, y) in BEV."""
    aux_outputs: InferenceAuxOutputs
    """Auxiliary outputs."""


# ==========================
# App + model initialization
# ==========================

app = FastAPI(title="DiffusionDrive NeuroNCAP Server", version="1.0.0")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_runner = DiffusionDriveRunner(CFG_PATH, CKPT_PATH, _device)

# Track temp image files for cleanup
_tmp_files: List[str] = []


# ==========================
# Helper functions
# ==========================

def _bytestr_to_numpy(pngs: List[bytes]) -> np.ndarray:
    """
    Convert a list of bytes (torch-saved tensors) to a numpy array (N, H, W, C).

    This mirrors the UniAD NeuroNCAP server:
      - upstream uses torch.save(tensor, buffer)
      - we recover with torch.load(io.BytesIO(...))
    """
    imgs = []
    for blob in pngs:
        img_t = torch.load(io.BytesIO(blob)).clone()  # (H, W, C) or (C, H, W)
        arr = img_t.numpy()
        # Normalize to HWC for PIL
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
            # assume CHW -> convert to HWC
            arr = np.moveaxis(arr, 0, -1)
        imgs.append(arr)
    return np.stack(imgs, axis=0)


def _save_numpy_images_to_temp(imgs: np.ndarray) -> List[str]:
    """
    Save N HWC uint8 images to temp PNG files and return their paths.
    """
    paths: List[str] = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        # Ensure uint8
        if img.dtype != np.uint8:
            img_clipped = np.clip(img, 0, 255)
            img = img_clipped.astype(np.uint8)
        pil_img = Image.fromarray(img)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pil_img.save(tmp, format="PNG")
        tmp.flush()
        tmp.close()
        paths.append(tmp.name)
        _tmp_files.append(tmp.name)
    return paths


def _np44(v: List[List[float]]) -> np.ndarray:
    A = np.array(v, dtype=np.float32)
    if A.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {A.shape}")
    return A


def _K_to_3x3(K: List[List[float]]) -> np.ndarray:
    arr = np.array(K, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :3]
    raise ValueError(f"camera intrinsic must be 3x3 or 4x4, got {arr.shape}")


def _build_lidar2img(calib: Calibration) -> (np.ndarray, np.ndarray):
    """
    Build lidar2img (N_cams, 4, 4) and cam_intrinsic (N_cams, 3, 3)
    from camera2image, camera2ego, lidar2ego.
    """
    l2e = _np44(calib.lidar2ego)
    lidar2imgs = []
    cam_intrinsics = []

    for cam in CAMERA_ORDER:
        if cam not in calib.camera2image or cam not in calib.camera2ego:
            raise HTTPException(status_code=400, detail=f"Missing calibration for camera {cam}")

        c2e = _np44(calib.camera2ego[cam])
        K3 = _K_to_3x3(calib.camera2image[cam])

        # 4x4 intrinsic
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K3

        e2c = np.linalg.inv(c2e).astype(np.float32)
        l2c = e2c @ l2e
        l2img = K4 @ l2c

        lidar2imgs.append(l2img.astype(np.float32))
        cam_intrinsics.append(K3.astype(np.float32))

    return np.stack(lidar2imgs, axis=0), np.stack(cam_intrinsics, axis=0)


def _build_lidar_pose(ego2world: List[List[float]], lidar2ego: List[List[float]]) -> np.ndarray:
    """
    lidar_pose = lidar2global = ego2world @ lidar2ego
    """
    E2W = _np44(ego2world)
    L2E = _np44(lidar2ego)
    return (E2W @ L2E).astype(np.float32)


def _build_ego_status(canbus: Any) -> np.ndarray:
    """
    Map CAN bus / ego-state into DiffusionDrive's 10D ego_status:

        [ accel(3), rotation_rate(3), vel(3), steer_deg(1) ]
    """
    ego = np.zeros(10, dtype=np.float32)

    # Dict-style CAN: explicit semantic keys
    if isinstance(canbus, dict):
        required_keys = ("accel", "rotation_rate", "vel")
        missing = [k for k in required_keys if k not in canbus]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"canbus dict missing keys: {missing}"
            )
        accel = np.array(canbus["accel"], dtype=np.float32).reshape(3)
        rot   = np.array(canbus["rotation_rate"], dtype=np.float32).reshape(3)
        vel   = np.array(canbus["vel"], dtype=np.float32).reshape(3)
        steer_deg = float(canbus.get("steer_deg", 0.0))

        ego[0:3] = accel
        ego[3:6] = rot
        ego[6:9] = vel
        ego[9]   = steer_deg
        return ego

    # List-style CAN: assume correct ordering
    vec = np.array(canbus, dtype=np.float32).reshape(-1)
    if vec.shape[0] < 10:
        raise HTTPException(
            status_code=400,
            detail=f"canbus list must have at least 10 elements, got {vec.shape[0]}"
        )

    ego[0:3] = vec[0:3]   # accel
    ego[3:6] = vec[3:6]   # rotation_rate
    ego[6:9] = vec[6:9]   # vel
    ego[9]   = float(vec[9])  # steer_deg
    return ego


# ==========================
# Endpoints
# ==========================

@app.get("/alive")
async def alive() -> bool:
    """Health check endpoint used by NeuroNCAP."""
    return True


@app.post("/reset")
async def reset_runner() -> bool:
    """
    Reset model temporal state and clean up temporary image files.
    Should be called at the start of each new scenario / sequence.
    """
    _runner.reset()

    # Clean up temp image files
    for p in _tmp_files[:]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        finally:
            try:
                _tmp_files.remove(p)
            except ValueError:
                pass

    return True


@app.post("/infer", response_model=InferenceOutputs)
async def infer(data: InferenceInputs) -> InferenceOutputs:
    # 1) Decode images in NuScenes camera order
    missing = [cam for cam in CAMERA_ORDER if cam not in data.images]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing images for cameras: {missing}"
        )

    img_blobs = [data.images[cam] for cam in CAMERA_ORDER]
    imgs_np = _bytestr_to_numpy(img_blobs)  # (6, H, W, C)
    img_paths = _save_numpy_images_to_temp(imgs_np)

    # 2) Build geometry: lidar2img, lidar_pose, cam_intrinsic
    try:
        lidar2img, cam_intrinsic = _build_lidar2img(data.calibration)
        lidar_pose = _build_lidar_pose(data.ego2world, data.calibration.lidar2ego)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3) Ego status + timestamp
    ego_status = _build_ego_status(data.canbus)
    ts_sec = float(data.timestamp) / 1e6  # microseconds -> seconds

    # 4) Build runner input
    dd_input = DiffusionDriveInferenceInput(
        img_paths=img_paths,
        lidar_pose=lidar_pose,
        lidar2img=lidar2img,
        ego_status=ego_status,
        timestamp=ts_sec,
        cam_intrinsic=cam_intrinsic,
    )

    # 5) Run inference (command: 0/1/2 or None)
    out = _runner.forward_inference(
        dd_input,
        new_scene=False,  # scene boundaries handled via /reset
        command=int(data.command) if data.command is not None else None,
    )

    # 6) Pack outputs using the NeuroNCAP/UniAD-compatible packer
    packed = pack_outputs_for_neuroncap(
        inference_output=out,
        aux=out.aux_outputs,
        class_names=getattr(_runner, "classes", None),
    )

    # packed = { "trajectory": 6x2 list, "aux_outputs": {...} }
    return InferenceOutputs(
        trajectory=packed["trajectory"],
        aux_outputs=InferenceAuxOutputs(**packed["aux_outputs"]),
    )


if __name__ == "__main__":
    # For local debugging;
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
