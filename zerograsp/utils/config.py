import argparse

import yaml


def parse_config(config_file_path=None):
    parser = argparse.ArgumentParser('Train a network for 3D reconstruction from a single stereo image.')
    parser.add_argument('--config', default='configs/mirage/config.yaml', help='config file')

    # General parameters
    parser.add_argument('--project_name', type=str, default='mirage')
    parser.add_argument('--model_name', type=str, default='mirage')
    parser.add_argument('--run_name', type=str, help='Run name of WandB')
    parser.add_argument('--train_dataset_name', type=str, default='mirage', help='Evaluation dataset name')
    parser.add_argument('--val_dataset_name', type=str, default='mirage', help='Validation dataset name')
    parser.add_argument('--eval_dataset_name', type=str, default=None, help='Evaluation dataset name')

    # Training parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file')
    parser.add_argument('--train_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/wds_graspnet_fix_graspness/shard-{000000..009999}.tar', help='URL to a webdataset for training')
    parser.add_argument('--val_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/wds_graspnet_fix_graspness/shard-{010000..010001}.tar', help='URL to a webdataset for validation')
    # parser.add_argument('--val_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/eval_datasets/woven_hard/shard-{000000..000003}.tar', help='URL to a webdataset for validation')
    parser.add_argument('--train_dataset_size', type=int, default=1000000)
    parser.add_argument('--val_dataset_size', type=int, default=200)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scheduler_step', type=int, default=3000)
    parser.add_argument('--scheduler_decay', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--mode',
                        type=str,
                        choices=['training', 'overfitting', 'validation', 'test', 'training_viz', 'validation_viz'])
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--valid_frame_name', type=str, default='valid_frames')

    # General Network parameters
    parser.add_argument('--backbone_model', type=str, default=None, choices=['dinov2', 'resnext', 'featup_dinov2', None])
    parser.add_argument('--input_feature_type', type=str, default='F', choices=['F', 'P', 'L', 'FL'])
    parser.add_argument('--single_obj', default=False, action='store_true', help='Enable the demo mode')
    parser.add_argument('--predict_grasp', default=False, action='store_true', help='Enable grasp prediction')

    # For demo
    parser.add_argument('--img_path', type=str, help='Image path for demo')
    parser.add_argument('--depth_path', type=str, help='Depth map path for demo')
    parser.add_argument('--mask_path', type=str, help='Mask path for demo')
    parser.add_argument('--camera_info_path', type=str, help='Camera info path for demo')

    # Mirage
    parser.add_argument('--depth_scale', type=float, default=0.1, help='Depth scale of depth images')
    parser.add_argument('--grid_size', type=float, default=0.2, help='Grid size in meter')
    parser.add_argument('--num_local_enc_layers', type=int, default=2, help='Number of layers for the local encoder')
    parser.add_argument('--num_enc_layers', type=int, default=2, help='Number of layers for PIVOT encoder')
    parser.add_argument('--num_dec_layers', type=int, default=2, help='Number of layers for PIVOT decoder')
    # parser.add_argument('--update_lod_freq', type=int, default=3, help='Frequency of updating LoD in epochs')
    parser.add_argument('--update_octree', default=False, action='store_true', help='Should update an octree for prediction?')
    parser.add_argument('--mode_sample', default=False, action='store_true', help='Should update an octree for prediction?')
    # parser.add_argument('--init_lod', type=int, default=6, help='Initial LoD')
    parser.add_argument('--max_lod', type=int, default=9, help='Number of maximum LoD')
    parser.add_argument('--min_lod', type=int, default=6, help='Number of maximum LoD')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads for multi-head attention (MHA)')
    parser.add_argument('--max_freq', type=int, default=32, help='Max freq (resolution) of positional encoding')
    parser.add_argument('--xpos_scale_base_denom', type=int, default=2, help='Denominator for xpos_scale_base')
    parser.add_argument('--pos_emb_dim', type=int, default=32, help='Positional embedding dimension for Transformer')
    parser.add_argument('--dim_model', type=int, default=32, help='Latent feature dimension for Backbone')
    parser.add_argument('--dim_mae', type=int, default=64, help='Latent feature dimension for Transformer')
    parser.add_argument('--resid_dropout', type=float, default=0.0, help='Dropout rate for MHA')
    parser.add_argument('--ff_dropout', type=float, default=0.1, help='Dropout rate for the feedforward network')
    parser.add_argument('--ff_activation',
                        type=str,
                        default='gelu',
                        help='Activation function for the feedforward network')
    parser.add_argument('--ff_hidden_layer_multiplier',
                        type=int,
                        default=4,
                        help='Hidden layer multiplier for the feedforward network')
    parser.add_argument('--head_mlp', type=str, default='siren', choices=['mlp', 'siren'])
    # parser.add_argument('--sampling_alg', type=str, default='surface', choices=['depth', 'surface'])
    # parser.add_argument('--sample_var', type=float, default=0.1, help='Variance for sampling points around surfaces')
    parser.add_argument('--pe_type', type=str, default='rope', choices=['wo', 'rope', 'cpe', 'ape', 'rpe'], help='Positional encoding type')
    parser.add_argument('--use_cpe', default=False, action='store_true', help='Use conditional positional encoding')
    parser.add_argument('--use_rpe', default=False, action='store_true', help='Use relative positional encoding')
    parser.add_argument('--use_rope', default=True, action='store_true', help='Use rotational positional encoding')
    parser.add_argument('--enc_patch_size', type=int, default=256, help='Size of patches')
    parser.add_argument('--dec_patch_size', type=int, default=256, help='Size of patches')
    parser.add_argument('--attn_type', type=str, default='self', choices=['self', 'cross'], help='Attention type')
    parser.add_argument('--mae_type', type=str, default='full', choices=['full', 'octree', 'dsa'], help='MAE type')
    parser.add_argument('--vot_scale_factor', type=int, default=2, help='Scale factor of an image for a voxel occlusion tester')
    parser.add_argument('--ofe_scale_factor', type=int, default=2, help='Scale factor of an image for an octree feature extractor')
    parser.add_argument('--kl_weight', type=float, default=10.0, help='Weight for KL divergence loss')
    parser.add_argument('--kl_loss_cycle_len', type=float, default=20000, help='Cycle length for KL divergence loss')
    # parser.add_argument('--use_mask3d', default=False, action='store_true', help='Use Mask3D for instance segmentation')
    # parser.add_argument('--use_mask3d_scheduler', default=False, action='store_true', help='Enable the demo mode')
    # parser.add_argument('--use_mask3d_rope', default=False, action='store_true', help='Enable the demo mode')
    # parser.add_argument('--mask3d_pred_disp', default=False, action='store_true', help='Enable the demo mode')
    parser.add_argument('--oneformer3d_dim', type=int, default=48, help='Latent feature dimension for Mask3D')
    parser.add_argument('--oneformer3d_num_heads', type=int, default=8, help='Number of heads for Mask3D multi-head attention')
    parser.add_argument('--oneformer3d_num_queries', type=int, default=25, help='Number of queries for Mask3D')
    parser.add_argument('--use_mult_obj_enc', default=False, action='store_true', help='Use multi-object encoding')
    parser.add_argument('--use_sing_obj_ref', default=False, action='store_true', help='Use single object refinement')
    parser.add_argument('--use_per_obj_latent', default=False, action='store_true', help='Use a per-object latent feature z')
    parser.add_argument('--concat_obj_latent', default=False, action='store_true', help='Concatenate features to compute z')
    parser.add_argument('--use_gt_depth', default=False, action='store_true', help='Use ground-truth depth maps for training')
    parser.add_argument('--use_full_layer', default=False, action='store_true', help='Use a full layer')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent feature dimension for the MLP head network')
    parser.add_argument('--ofe_use_conv3d', default=False, action='store_true', help='Use Conv3DEncoder for OFE')
    parser.add_argument('--use_pool_and_concat', default=False, action='store_true', help='Use a pool and concate strategy')
    parser.add_argument('--use_only_visible', default=False, action='store_true', help='Use only visible regions for grasp pose prediction')
    parser.add_argument('--use_aug', default=False, action='store_true', help='Use augmentation')
    parser.add_argument('--fine_tuning', default=False, action='store_true', help='Enable fine tuning (freeze an encoder)')
    parser.add_argument('--fine_tuning_decoder', default=False, action='store_true', help='Enable fine tuning (freeze an decoder)')
    parser.add_argument('--use_collision_constraints', default=False, action='store_true', help='Use collision constraint')
    parser.add_argument('--use_collision_detection', default=False, action='store_true', help='Use collision detector')
    parser.add_argument('--use_collision_detection_only_with_depth_map', default=False, action='store_true', help='Use collision detector')


    # Convolutional Occupancy Networks and StereoPiFu
    parser.add_argument('--latent_feature_dim',
                        type=int,
                        default=32,
                        help='Latent feature dimension for the MLP head network')
    parser.add_argument('--dist_norm_factor',
                        type=float,
                        default=5.0,
                        help='Normalization factor for a relative z distance')
    parser.add_argument('--use_sigmoid_dist',
                        default=False,
                        action='store_true',
                        help='Use a sigmoid distance to compute a relative z offset?')
    parser.add_argument('--use_align_corners', default=True, help='Use align_corners=True for grid sampling?')
    parser.add_argument('--use_normal_map', default=False, action='store_true', help='Use a normal map?')
    parser.add_argument('--use_pos_enc',
                        default=False,
                        action='store_true',
                        help='Use positional encoding for a relative z offset?')
    parser.add_argument('--use_sdf', default=False, action='store_true', help='Use sdf instead of occupancy maps?')
    parser.add_argument('--use_ray', default=False, action='store_true', help='Use ray-based representation?')
    parser.add_argument('--use_voxel', default=False, action='store_true', help='Use explicit voxel representation?')
    parser.add_argument('--grid_res', type=int, default=64, help='Resolution of a voxel grid')
    parser.add_argument('--use_ground_plane',
                        default=False,
                        action='store_true',
                        help='Use explicit ground plane representation?')
    parser.add_argument('--use_triplane',
                        default=False,
                        action='store_true',
                        help='Use explicit triplane representation?')
    parser.add_argument('--plane_res', type=int, default=256, help='Resolution of a ground plane')

    args, _ = parser.parse_known_args()
    args = vars(args)
    args_default = {k: parser.get_default(k) for k in args}
    if config_file_path is None:
        args_config = yaml.load(open(args['config']), Loader=yaml.FullLoader)
    else:
        args_config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    args_inline = {k: v for (k, v) in args.items() if v != args_default[k]}
    args = args_default.copy()
    args.update(args_config)
    args.update(args_inline)
    args = argparse.Namespace(**args)
    print(args)
    return args
