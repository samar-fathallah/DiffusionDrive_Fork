import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict,Any
import cv2
import os
import importlib
import numpy as np
import torch
from mmdet.datasets.pipelines import Compose
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet.models import build_detector


# ---- Data structures ----
def _build_test_aug_config(cfg):
    """
    Reproduce NuScenes3DDataset.get_augmentation() for test_mode=True.

    Pulls H, W, final_dim, bot_pct_lim from the same place the dataset does:
    cfg.data.test.data_aug_conf  (fallback: cfg.data.val.data_aug_conf)
    """
    # locate data_aug_conf
    data_aug_conf = None
    try:
        data_aug_conf = cfg.data.test.data_aug_conf
    except Exception:
        pass
    if data_aug_conf is None:
        try:
            data_aug_conf = cfg.data.val.data_aug_conf
        except Exception:
            pass

    # If no aug config is found, return None 
    if data_aug_conf is None:
        print("[WARN] No data_aug_conf found in cfg.data.test/val; skipping ResizeCropFlipImage.")
        return None

    # unpack the same fields used by the dataset
    H = data_aug_conf["H"]                 # e.g. 900
    W = data_aug_conf["W"]                 # e.g. 1600
    fH, fW = data_aug_conf["final_dim"]    # e.g. (256, 704)
    bot_pct_lim = data_aug_conf["bot_pct_lim"]  # e.g. (0.0, 0.0)

    # test-mode branch 
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))   # (newW, newH)
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(bot_pct_lim)) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    rotate_3d = 0

    aug_config = {
        "resize": resize,
        "resize_dims": resize_dims,  # (newW, newH)
        "crop": crop,                # (x1, y1, x2, y2)
        "flip": flip,
        "rotate": rotate,
        "rotate_3d": rotate_3d,
    }
    print("[build_test_aug_config] "
          f"H,W=({H},{W}) final_dim=({fH},{fW}) -> "
          f"resize={resize:.4f} resize_dims={resize_dims} crop={crop}")
    return aug_config


@dataclass
class DiffusionDriveInferenceInput:
    """
    Input to the DiffusionDrive runner (one frame).

    - img_paths: list of 6 image file paths (strings), in camera order
    - lidar_pose: (4, 4) lidar->global transform
    - lidar2img: (N_cams, 4, 4) projection matrices
    - ego_status: (10,) from get_ego_status()
    - timestamp: seconds (float)
    - cam_intrinsic: (N_cams, 3, 3), optional but recommended
    """
    img_paths: List[str]
    lidar_pose: np.ndarray
    lidar2img: np.ndarray
    ego_status: np.ndarray
    timestamp: float
    cam_intrinsic: Optional[np.ndarray] = None


@dataclass
class DiffusionDriveAuxOutputs:
    boxes_3d: Optional[np.ndarray] = None
    scores_3d: Optional[np.ndarray] = None
    labels_3d: Optional[np.ndarray] = None
    cls_scores: Optional[np.ndarray] = None
    instance_ids: Optional[np.ndarray] = None

    vectors: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    trajs_3d: Optional[np.ndarray] = None
    trajs_score: Optional[np.ndarray] = None
    anchor_queue: Optional[np.ndarray] = None
    period: Optional[np.ndarray] = None

    planning_score: Optional[np.ndarray] = None
    planning: Optional[np.ndarray] = None
    ego_period: Optional[np.ndarray] = None
    ego_anchor_queue: Optional[np.ndarray] = None

    def to_json(self) -> dict:
        def _to_list(x):
            return x.tolist() if x is not None else None

        return dict(
            boxes_3d=_to_list(self.boxes_3d),
            scores_3d=_to_list(self.scores_3d),
            labels_3d=_to_list(self.labels_3d),
            cls_scores=_to_list(self.cls_scores),
            instance_ids=_to_list(self.instance_ids),
            vectors=_to_list(self.vectors),
            scores=_to_list(self.scores),
            labels=_to_list(self.labels),
            trajs_3d=_to_list(self.trajs_3d),
            trajs_score=_to_list(self.trajs_score),
            anchor_queue=_to_list(self.anchor_queue),
            period=_to_list(self.period),
            planning_score=_to_list(self.planning_score),
            planning=_to_list(self.planning),
            ego_period=_to_list(self.ego_period),
            ego_anchor_queue=_to_list(self.ego_anchor_queue),
        )

    @classmethod
    def empty(cls) -> "DiffusionDriveAuxOutputs":
        return cls()


@dataclass
class DiffusionDriveInferenceOutput:
    trajectory: np.ndarray            # (6, 2) ego future trajectory
    aux_outputs: DiffusionDriveAuxOutputs


# ---- Helpers ----

def _maybe_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _strip_gt_from_collect(cfg):
    """
    Remove GT/label keys from the Collect step in cfg.data.test (and val if present),
    so pure inference doesn't require ground-truth fields.
    """
    def _process_pipeline(pipeline, tag):
        for i, step in enumerate(pipeline):
            if isinstance(step, dict) and step.get("type") == "Collect":
                keep = []
                drop_set = {
                    # planning GT
                    "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd",
                    # agent/map/det GT (safe to drop for inference)
                    "gt_agent_fut_trajs", "gt_agent_fut_masks",
                    "gt_bboxes_3d", "gt_labels_3d", "gt_depth",
                    "gt_map_labels", "gt_map_pts", "vectors",
                }
                orig_keys = step.get("keys", [])
                for k in orig_keys:
                    if k not in drop_set:
                        keep.append(k)
                step["keys"] = keep
                pipeline[i] = step
                print(f"[runner] Stripped GT keys from Collect in {tag}:")
                print(f"         kept keys = {keep}")
                break  # only one Collect in this pipeline

    if hasattr(cfg.data, "test") and hasattr(cfg.data.test, "pipeline"):
        _process_pipeline(cfg.data.test.pipeline, "data.test")
    if hasattr(cfg.data, "val") and hasattr(cfg.data.val, "pipeline"):
        _process_pipeline(cfg.data.val.pipeline, "data.val")

def _derive_cmd_from_ego_status(
    ego_status: np.ndarray,
    left_thresh_deg: float = 2.0,
    right_thresh_deg: float = -2.0,
) -> np.ndarray:
    """
    Derive a 3-way command [right, left, straight] from *degree* steering angle.

    ego_status layout: accel(3), rotation_rate(3), vel(3), steer(1) -> total 10
    steer angle index = 9 (in DEGREES).
    """
    try:
        steer_deg = float(ego_status[9])
    except Exception:
        steer_deg = 0.0

    if steer_deg >= left_thresh_deg:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Turn Left
    elif steer_deg <= right_thresh_deg:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Turn Right
    else:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Go Straight


# ---- Runner ----

class DiffusionDriveRunner:
    """
    Proper runner that:

    - loads DiffusionDrive (V1SparseDrive) from config + checkpoint
    - uses the FULL test pipeline (including LoadMultiViewImageFromFiles)
    - expects image *paths* (img_paths), not raw numpy images
    - mimics single-sample DataLoader + collate behavior
    - injects gt_ego_fut_cmd derived from ego_status (NeuroNCAP-compatible)
    - returns final ego planning trajectory + aux outputs
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        # 1) Load config
        cfg = Config.fromfile(config_path)
        self.cfg = cfg
        self._override_cmd_one_hot = None
        # 2) custom_imports if present
        if cfg.get("custom_imports", None):
            import_modules_from_strings(**cfg["custom_imports"])

        # 3) Handle plugin / plugin_dir (registers custom pipelines, heads, etc.)
        if hasattr(cfg, "plugin") and cfg.plugin:
            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir  # e.g. "projects/mmdet3d_plugin/"
                _module_dir = os.path.dirname(plugin_dir)
            else:
                # fallback: directory of the config file
                _module_dir = os.path.dirname(config_path)

            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print("Importing plugin module:", _module_path)
            importlib.import_module(_module_path)

        # 4) Build test pipeline (strip GT from Collect for inference)
        _strip_gt_from_collect(self.cfg)
        test_pipeline_cfg = cfg.data.test.pipeline
        self.pipeline = Compose(test_pipeline_cfg)

        # 5) Build model
        cfg.model.train_cfg = None
        self.model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

        # 6) Load checkpoint
        if checkpoint_path is not None:
            ckpt = load_checkpoint(self.model, checkpoint_path, map_location="cpu")
            if "CLASSES" in ckpt.get("meta", {}):
                self.classes = ckpt["meta"]["CLASSES"]
            else:
                self.classes = getattr(self.model, "CLASSES", None)
        else:
            raise ValueError("checkpoint_path must not be None")

        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        # 7) Scene / temporal state
        self.scene_token = str(uuid.uuid4())
        self.reset()

    def reset(self):
        """Reset per-scene temporal state (queues, banks, etc.)."""
        # New scene token for logging / potential future use
        old_token = getattr(self, "scene_token", None)
        self.scene_token = str(uuid.uuid4())
        print(f"[runner] Scene reset. old_scene_token={old_token}, new_scene_token={self.scene_token}")

        # ---- Reset detection / map instance banks ----
        head = getattr(self.model, "head", None)
        if head is not None:
            # Detection bank
            det_head = getattr(head, "det_head", None)
            if det_head is not None and hasattr(det_head, "instance_bank"):
                bank = det_head.instance_bank
                if hasattr(bank, "reset"):
                    print("[runner] Resetting det_head.instance_bank")
                    bank.reset()

            # Map bank (if map_head is enabled)
            map_head = getattr(head, "map_head", None)
            if map_head is not None and hasattr(map_head, "instance_bank"):
                bank = map_head.instance_bank
                if hasattr(bank, "reset"):
                    print("[runner] Resetting map_head.instance_bank")
                    bank.reset()

            # ---- Reset motion/planning instance queue ----
            motion_plan_head = getattr(head, "motion_plan_head", None)
            if motion_plan_head is not None and hasattr(motion_plan_head, "instance_queue"):
                q = motion_plan_head.instance_queue
                if hasattr(q, "reset"):
                    print("[runner] Resetting motion_plan_head.instance_queue")
                    q.reset()


    def set_command_override(self, command_int: Optional[int] = None, one_hot: Optional[np.ndarray] = None):
        if one_hot is not None:
            v = np.asarray(one_hot, dtype=np.float32).reshape(3)
        elif command_int is not None:
            if command_int == 0:   v = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # right
            elif command_int == 1: v = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # left
            else:                  v = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # straight
        else:
            v = None
        self._override_cmd_one_hot = v
    @torch.no_grad()
    def forward_inference(self, inp: DiffusionDriveInferenceInput, new_scene: bool = False,
                      command: Optional[int] = None):
        """
        Run inference with the full MMDet-style preprocessing pipeline.

        Pipeline from config:

        - LoadMultiViewImageFromFiles
        - ResizeCropFlipImage
        - NormalizeMultiviewImage
        - NuScenesSparse4DAdaptor
        - Collect(...)
        """

        # -----------------------------
        # 1) Prepare "dataset-style" dict for the pipeline
        # -----------------------------
        if new_scene:
            print("[runner] NEW SCENE FLAG RECEIVED → resetting temporal state")
            self.reset()
        num_cams = len(inp.img_paths)
        # --- Camera order enforcement ---------------------------------------
        expected_order = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT",
                        "BACK", "BACK_LEFT", "BACK_RIGHT"]

        # Only warn (don’t enforce file naming convention)
        for i, p in enumerate(inp.img_paths):
            pname = p.lower()
            if "front" in pname and i != 0:
                print("[WARN] CAM_FRONT expected at index 0 but path appears elsewhere:", p)
            if "front_right" in pname and i != 1:
                print("[WARN] CAM_FRONT_RIGHT expected at index 1:", p)
            if "front_left" in pname and i != 2:
                print("[WARN] CAM_FRONT_LEFT expected at index 2:", p)
            if "back" in pname and "back_left" not in pname and "back_right" not in pname and i != 3:
                print("[WARN] CAM_BACK expected at index 3:", p)

        assert inp.lidar2img.shape[0] == num_cams, (
            f"lidar2img first dim {inp.lidar2img.shape[0]} != num_cams {num_cams}"
        )

        input_dict = dict(
            img_filename=list(inp.img_paths),
            lidar2img=[m.astype(np.float32) for m in inp.lidar2img],  # list of (4,4)
            lidar2global=inp.lidar_pose.astype(np.float32),           # (4,4)
            timestamp=inp.timestamp,                                  # float (sec)
            ego_status=inp.ego_status.astype(np.float32),             # (10,)
        )
        if inp.cam_intrinsic is not None:
            input_dict["cam_intrinsic"] = [K.astype(np.float32) for K in inp.cam_intrinsic]
        # --- Validate shapes --------------------------------------------------
        if not (isinstance(inp.ego_status, np.ndarray) and inp.ego_status.shape == (10,)):
            print("[WARN] ego_status shape expected (10,), got", inp.ego_status.shape)

        if inp.lidar2img.shape[1:] != (4, 4):
            print("[WARN] lidar2img each must be (4,4), got", inp.lidar2img.shape)

        if inp.cam_intrinsic is not None and inp.cam_intrinsic.shape != (num_cams, 3, 3):
            print("[WARN] cam_intrinsic expected (6,3,3), got", inp.cam_intrinsic.shape)

        # -----------------------------
        # 2) Run through pipeline
        # -----------------------------
        aug_config = _build_test_aug_config(self.cfg)  # exact test-mode augmentation
        # Warn if resolution mismatch
        if aug_config is not None:
            H_cfg, W_cfg = self.cfg.data.test.data_aug_conf["H"], self.cfg.data.test.data_aug_conf["W"]
            if (H_cfg, W_cfg) != (900, 1600):
                print("[WARN] Training expects 1600x900 raw images — input may differ")

        input_dict["aug_config"] = aug_config          # makes ResizeCropFlipImage run
        if aug_config is None:
            print("[runner] aug_config=None -> ResizeCropFlipImage will be skipped.")
        else:
            print("[runner] aug_config set:", aug_config)
        
        # PATCH: AUTO-SCALE lidar2img IF IMAGE RESOLUTION CHANGED
        # ------------------------------------------------------------
        # Expected raw resolution for DiffusionDrive (NuScenes)
        EXPECTED_W = 1600
        EXPECTED_H = 900
        try:
            sample_img = cv2.imread(inp.img_paths[0])
            if sample_img is None:
                raise ValueError("cv2.imread returned None")
            H_in, W_in = sample_img.shape[:2]
        except Exception as e:
            print("[WARN] Could not load temp image for resolution detection:", e)
            H_in, W_in = EXPECTED_H, EXPECTED_W  # fallback to expected size

        # If resolution differs, need scaling
        if (H_in, W_in) != (EXPECTED_H, EXPECTED_W):
            print(f"[runner][INFO] Image resolution changed: got {(H_in, W_in)}, expected {(EXPECTED_H, EXPECTED_W)}")

            # Compute scale factors
            sx = W_in / EXPECTED_W
            sy = H_in / EXPECTED_H

            print(f"[runner][INFO] Scaling intrinsics by factors: sx={sx:.4f}, sy={sy:.4f}")

            # Scale lidar2img
            scaled_lidar2img = inp.lidar2img.copy()
            for i in range(scaled_lidar2img.shape[0]):
                # Scale camera intrinsic components:
                # lidar2img = K * [R|t] so we scale K
                scaled_lidar2img[i, 0, :] *= sx     # fx and cx scale with width
                scaled_lidar2img[i, 1, :] *= sy     # fy and cy scale with height

            # Replace with scaled version
            input_dict["lidar2img"] = [m.astype(np.float32) for m in scaled_lidar2img]

        else:
            # Resolution matches — no scaling needed
            input_dict["lidar2img"] = [m.astype(np.float32) for m in inp.lidar2img]

        data = self.pipeline(input_dict)  # single-sample dict (pre-collate)

        # -----------------------------
        # 3) Unpack DataContainer & create batch dimension (bs=1)
        # -----------------------------
        assert isinstance(data["img"], DataContainer)
        img = data["img"].data.to(self.device)   # (N_cams, 3, H, W)
        img = img.unsqueeze(0)                   # (1, N_cams, 3, H, W)

        assert isinstance(data["img_metas"], DataContainer)
        img_metas = data["img_metas"].data    
        # normalize to list like collate_fn would do
        if not isinstance(img_metas, (list, tuple)):
            img_metas = [img_metas]

        try:
            print("[runner] img tensor shape:", data["img"].data.shape)  # (6, 3, h, w)
            print("[runner] img_metas len:", len(img_metas))
            print("[runner] img_metas[0] keys:", list(img_metas[0].keys()))
        except Exception as _e:
            print("[runner] meta print skipped:", repr(_e))

        def _get_tensor(name, add_batch_dim=True):
            if name not in data:
                return None
            val = data[name]
            if isinstance(val, DataContainer):
                val = val.data
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)
                if add_batch_dim and val.dim() >= 1:
                    val = val.unsqueeze(0)
            return val

        timestamp_t      = _get_tensor("timestamp", add_batch_dim=False)
        projection_mat_t = _get_tensor("projection_mat", add_batch_dim=True)
        image_wh_t       = _get_tensor("image_wh", add_batch_dim=True)
        ego_status_t     = _get_tensor("ego_status", add_batch_dim=True)

        # -----------------------------
        # 3.5) Derive & inject gt_ego_fut_cmd for inference
        # -----------------------------
        # --- Command handling -------------------------------------------------
        if command is not None:
            # NeuroNCAP passed explicit command (0=R,1=L,2=S)
            if command == 0:
                cmd_np = np.array([1,0,0], dtype=np.float32)
            elif command == 1:
                cmd_np = np.array([0,1,0], dtype=np.float32)
            else:
                cmd_np = np.array([0,0,1], dtype=np.float32)
            print("[runner] using command from API:", command, cmd_np.tolist())

        elif getattr(self, "_override_cmd_one_hot", None) is not None:
            cmd_np = self._override_cmd_one_hot.astype(np.float32)
            print("[runner] using OVERRIDDEN gt_ego_fut_cmd:", cmd_np.tolist())
            self._override_cmd_one_hot = None  # consume override

        else:
            # Default: derive from steering angle
            cmd_np = _derive_cmd_from_ego_status(inp.ego_status)
            print("[runner] injected gt_ego_fut_cmd (from steer):", cmd_np.tolist())


        gt_ego_fut_cmd_t = torch.from_numpy(cmd_np).unsqueeze(0).to(self.device)
        if image_wh_t is not None:
            print("[runner] image_wh shape:", tuple(image_wh_t.shape))       # expect (1, 6, 2)
        else:
            print("[runner] image_wh: None")
        if projection_mat_t is not None:
            print("[runner] projection_mat shape:", tuple(projection_mat_t.shape))  # expect (1, 6, 4, 4)
        else:
            print("[runner] projection_mat: None")

        # -----------------------------
        # 4) Call model.simple_test
        # -----------------------------
        data_kwargs = dict(img_metas=img_metas,
                           gt_ego_fut_cmd=gt_ego_fut_cmd_t)  # <-- inject here
        if timestamp_t is not None:      data_kwargs["timestamp"]      = timestamp_t
        if projection_mat_t is not None: data_kwargs["projection_mat"] = projection_mat_t
        if image_wh_t is not None:       data_kwargs["image_wh"]       = image_wh_t
        if ego_status_t is not None:     data_kwargs["ego_status"]     = ego_status_t
        try:
            steer_local = float(inp.ego_status[9])
        except Exception:
            steer_local = float('nan')
        vel_local = inp.ego_status[6:9] if inp.ego_status.size >= 9 else np.zeros(3, dtype=np.float32)
        spd_local = float(np.linalg.norm(vel_local))
        print(f"[runner] steer(deg)={steer_local:.3f}  vel(ego)={vel_local.tolist()}  speed={spd_local:.3f} m/s")
        if spd_local < 0.1:
            print("[runner][WARN] speed is ~0; plan may be ~0 displacement")

        outputs = self.model.simple_test(img, **data_kwargs)
        # Right after model.simple_test(...)
        img_bbox = outputs[0]["img_bbox"]
        boxes_3d = img_bbox.get("boxes_3d", None)

        try:
            print("boxes_3d type:", type(boxes_3d))
            if hasattr(boxes_3d, "__class__"):
                print("boxes_3d class name:", boxes_3d.__class__.__name__)
        except Exception as e:
            print("boxes_3d type check failed:", e)

        # Also check metas if present
        if isinstance(img_metas, list) and img_metas:
            bt = img_metas[0].get("box_type_3d", None)
            if bt is not None:
                print("img_metas[0]['box_type_3d']:", bt)


        # -----------------------------
        # 5) Extract final ego planning trajectory
        # -----------------------------
        final_planning = img_bbox["final_planning"]  # Tensor (6,2)
        trajectory = final_planning.detach().cpu().numpy()

        # -----------------------------
        # 6) Pack auxiliary outputs
        # -----------------------------
        aux = DiffusionDriveAuxOutputs(
            boxes_3d=_maybe_to_numpy(img_bbox.get("boxes_3d")),
            scores_3d=_maybe_to_numpy(img_bbox.get("scores_3d")),
            labels_3d=_maybe_to_numpy(img_bbox.get("labels_3d")),
            cls_scores=_maybe_to_numpy(img_bbox.get("cls_scores")),
            instance_ids=_maybe_to_numpy(img_bbox.get("instance_ids")),
            vectors=_maybe_to_numpy(img_bbox.get("vectors")),
            scores=_maybe_to_numpy(img_bbox.get("scores")),
            labels=_maybe_to_numpy(img_bbox.get("labels")),
            trajs_3d=_maybe_to_numpy(img_bbox.get("trajs_3d")),
            trajs_score=_maybe_to_numpy(img_bbox.get("trajs_score")),
            anchor_queue=_maybe_to_numpy(img_bbox.get("anchor_queue")),
            period=_maybe_to_numpy(img_bbox.get("period")),
            planning_score=_maybe_to_numpy(img_bbox.get("planning_score")),
            planning=_maybe_to_numpy(img_bbox.get("planning")),
            ego_period=_maybe_to_numpy(img_bbox.get("ego_period")),
            ego_anchor_queue=_maybe_to_numpy(img_bbox.get("ego_anchor_queue")),
        )

        return DiffusionDriveInferenceOutput(
            trajectory=trajectory,
            aux_outputs=aux,
        )


# ===================== NeuroNCAP packer (UniAD-compatible) =====================
# If it uses y-front, x-right (old style), set this to "legacy".
DIFFUSIONDRIVE_COORD_FRAME = "modern"  # "modern" (x-front,y-left) or "legacy"

def _maybe_swap_xy_inplace(arr: np.ndarray) -> None:
    """In-place swap for Traj [x,y] if you ever need old->new frame; default is NO-OP."""
    x = arr[..., 0].copy()
    y = arr[..., 1].copy()
    # legacy -> modern: x(front)=y_legacy, y(left)=-x_legacy
    arr[..., 0] = y
    arr[..., 1] = -x

def _format_boxes_for_bev(
    boxes_3d: Optional[np.ndarray],
    label_ids: Optional[np.ndarray],
    scores_3d: Optional[np.ndarray],
    instance_ids: Optional[np.ndarray],
    class_names: Optional[List[str]],
) -> Dict[str, Optional[List]]:
    """
    Convert DiffusionDrive boxes into NeuroNCAP/UniAD's BEV schema:
    objects_in_bev: [x, y, width, length, yaw]
    """
    if boxes_3d is None or len(boxes_3d) == 0:
        return dict(
            objects_in_bev=None,
            object_classes=None,
            object_scores=None,
            object_ids=None,
        )

    # Common mmdet3d layout: [x, y, z, dx, dy, dz, yaw, ...]
    objects_in_bev = []
    for b in boxes_3d:
        x, y, _, dx, dy, _, yaw = b[:7]
        # width = dy (left-right), length = dx (front-back)
        objects_in_bev.append([float(x), float(y), float(dy), float(dx), float(yaw)])

    if DIFFUSIONDRIVE_COORD_FRAME == "legacy":
        # convert each [x,y,*,*,yaw] (only xy need change)
        objs_np = np.array(objects_in_bev, dtype=np.float32)
        _maybe_swap_xy_inplace(objs_np[..., :2])
        objects_in_bev = objs_np.tolist()

    object_scores = [float(s) for s in scores_3d] if scores_3d is not None else None
    object_ids    = [int(i) for i in instance_ids] if instance_ids is not None else None

    if label_ids is not None:
        if class_names:
            object_classes = [
                class_names[int(l)] if 0 <= int(l) < len(class_names) else str(int(l))
                for l in label_ids
            ]
        else:
            object_classes = [str(int(l)) for l in label_ids]
    else:
        object_classes = None

    return dict(
        objects_in_bev=objects_in_bev,
        object_classes=object_classes,
        object_scores=object_scores,
        object_ids=object_ids,
    )

def _format_future_trajs_for_api(trajs_3d: Optional[np.ndarray]) -> Optional[List]:
    """
    Pass-through N x 6 x 12 x 2 (object forecasts) if present.
    If legacy coords, convert to modern (x-front,y-left).
    """
    if trajs_3d is None or len(trajs_3d) == 0:
        return None
    trajs = np.array(trajs_3d, dtype=np.float32)
    if DIFFUSIONDRIVE_COORD_FRAME == "legacy":
        _maybe_swap_xy_inplace(trajs[..., :2])
    return trajs.tolist()

def _format_ego_traj_for_api(ego_traj: np.ndarray) -> List[List[float]]:
    """
    ego_traj: (6,2) numpy -> list; convert if legacy coords.
    """
    traj = np.array(ego_traj, dtype=np.float32)
    if DIFFUSIONDRIVE_COORD_FRAME == "legacy":
        _maybe_swap_xy_inplace(traj)
    return traj.tolist()

def pack_outputs_for_neuroncap(
    inference_output: "DiffusionDriveInferenceOutput",
    aux: Optional["DiffusionDriveAuxOutputs"] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert runner output to EXACT NeuroNCAP/UniAD schema:
      {
        "trajectory": 6x2 list,
        "aux_outputs": {
          "objects_in_bev": Nx5 or None,
          "object_classes": list or None,
          "object_scores": list or None,
          "object_ids": list or None,
          "future_trajs": N x 6 x 12 x 2 or None
        }
      }
    """
    if aux is None:
        aux = inference_output.aux_outputs

    aux_json = aux.to_json()

    det = _format_boxes_for_bev(
        boxes_3d=aux_json.get("boxes_3d"),
        label_ids=aux_json.get("labels_3d"),
        scores_3d=aux_json.get("scores_3d"),
        instance_ids=aux_json.get("instance_ids"),
        class_names=class_names,
    )

    packed = dict(
        trajectory=_format_ego_traj_for_api(inference_output.trajectory),
        aux_outputs=dict(
            objects_in_bev=det["objects_in_bev"],
            object_classes=det["object_classes"],
            object_scores=det["object_scores"],
            object_ids=det["object_ids"],
            future_trajs=_format_future_trajs_for_api(aux_json.get("trajs_3d")),
        ),
    )
    return packed
