######################################################################
# DiffusionDrive (V1SparseDrive) — CLEAN INFERENCE CONFIG (FIXED)
######################################################################

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

#########################
# BASIC GLOBAL SETTINGS #
#########################

input_shape = (704, 256)   # (W, H)
embed_dims = 256
num_groups = 8
drop_out = 0.1

# Strides and FPN levels
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
use_deformable_func = True

# Temporal flags
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

# Decoder counts
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1

##########
# CLASSES
##########

class_names = [
    "car","truck","construction_vehicle","bus","trailer","barrier",
    "motorcycle","bicycle","pedestrian","traffic_cone",
]
map_class_names = ["ped_crossing","divider","boundary"]

num_classes = len(class_names)
num_map_classes = len(map_class_names)

###############
# KEY VARS
###############

roi_size = (30, 60)
num_sample = 20

fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,
)

############################
# MODEL (original complete)
############################

model = dict(
    type="V1SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,

    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),

    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256,512,1024,2048],
    ),

    depth_branch=dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),

    head=dict(
        type="V1SparseDriveHead",
        task_config=task_config,

        #############################################################
        # DETECTION HEAD (unchanged from original)
        #############################################################
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,

            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor="data/kmeans/kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),

            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128,32,32,64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),

            num_single_frame_decoder=num_single_frame_decoder,

            operation_order=(
                [
                    "gnn","norm","deformable","ffn","norm","refine",
                ] * num_single_frame_decoder
                + [
                    "temp_gnn","gnn","norm","deformable","ffn","norm","refine",
                ] * (num_decoder - num_single_frame_decoder)
            )[2:],

            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims*2 if decouple_attn else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ) if temporal else None,

            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims*2 if decouple_attn else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims*2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0,0,0],
                        [0.45,0,0],
                        [-0.45,0,0],
                        [0,0.45,0],
                        [0,-0.45,0],
                        [0,0,0.45],
                        [0,0,-0.45],
                    ],
                ),
            ),

            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),

            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2]*3 + [0.5]*7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2]*3 + [0.5]*3 + [0]*4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [2,2,2,1,1,1,0,0,1,1],
                },
            ),

            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),

            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),

            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2]*3 + [1]*7,
        ),

        #############################################################
        # MAP HEAD (unchanged)
        #############################################################
        map_head= dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,

            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="data/kmeans/kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=33 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),

            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),

            num_single_frame_decoder=num_single_frame_decoder_map,

            operation_order=(
                [
                    "gnn","norm","deformable","ffn","norm","refine",
                ] * num_single_frame_decoder_map
                + [
                    "temp_gnn","gnn","norm","deformable","ffn","norm","refine",
                ] * (num_decoder - num_single_frame_decoder_map)
            ),

            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims*2 if decouple_attn_map else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ) if temporal_map else None,

            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims*2 if decouple_attn_map else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims*2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0,0.5,-0.5,1,-1),
                    ground_height=-1.84,
                ),
            ),

            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),

            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type="HungarianLinesAssigner",
                    cost=dict(
                        type="MapQueriesCost",
                        cls_cost=dict(type="FocalLossCost", weight=1.0),
                        reg_cost=dict(type="LinesL1Cost", weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),

            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),

            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type="LinesL1Loss",
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),

            decoder=dict(type="SparsePoint3DDecoder"),

            reg_weights=[1.0]*40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix="map",
        ),

        #############################################################
        # MOTION + PLANNING HEAD (unchanged)
        #############################################################
        motion_plan_head=dict(
            type="V13MotionPlanningHead",
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            if_init_timemlp=False,

            motion_anchor=f"data/kmeans/kmeans_motion_{fut_mode}.npy",
            plan_anchor=f"data/kmeans/kmeans_plan_{ego_fut_mode}.npy",

            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,

            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.2,
                feature_map_scale=(input_shape[1]/strides[-1],
                                   input_shape[0]/strides[-1]),
            ),

            interact_operation_order=(
                ["temp_gnn","gnn","norm","cross_gnn","norm","ffn","norm"] * 3
                + ["refine"]
            ),

            diff_operation_order=(
                [
                    "traj_pooler","self_attn","norm",
                    "agent_cross_gnn","norm",
                    "anchor_cross_gnn","norm",
                    "ffn","norm","modulation","diff_refine",
                ] * 2
            ),

            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims*2 if decouple_attn_motion else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims*2 if decouple_attn_motion else embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            cross_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            self_attn_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),

            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            refine_layer=dict(
                type="V11MotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),

            diff_refine_layer=dict(
                type="V4DiffMotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                if_zeroinit_reg=False,
            ),

            modulation_layer=dict(
                type="V1ModulationLayer",
                embed_dims=embed_dims,
                if_global_cond=False,
                if_zeroinit_scale=False,
            ),

            traj_pooler_layer=dict(
                type="V1TrajPooler",
                embed_dims=embed_dims,
                ego_fut_ts=ego_fut_ts,
            ),

            motion_sampler=dict(type="MotionTarget"),

            motion_loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2,
            ),

            motion_loss_reg=dict(type="L1Loss", loss_weight=0.2),

            planning_sampler=dict(
                type="V1PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),

            plan_loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            ),

            plan_loss_reg=dict(type="L1Loss", loss_weight=1.0),

            plan_loss_status=dict(type="L1Loss", loss_weight=1.0),

            motion_decoder=dict(type="SparseBox3DMotionDecoder"),

            planning_decoder=dict(
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                use_rescore=True,
            ),

            num_det=50,
            num_map=10,
        ),
    ),
)

###########################################
# TEST PIPELINE (minimal)
###########################################

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img","timestamp","projection_mat","image_wh","ego_status",
            "gt_ego_fut_trajs","gt_ego_fut_masks","gt_ego_fut_cmd",
        ],
        meta_keys=["T_global","T_global_inv","timestamp"],
    ),
]

###########################################
# DATA (test only)
###########################################

data = dict(
    test=dict(
        type="NuScenes3DDataset",
        data_root="data/nuscenes/",
        ann_file="data/infos/nuscenes_infos_val.pkl",
        classes=class_names,
        modality=dict(use_camera=True),
        pipeline=test_pipeline,
        data_aug_conf=dict(
            H=900, W=1600,
            final_dim=input_shape[::-1],
            bot_pct_lim=(0.0, 0.0),
        ),
        test_mode=True,
    )
)
