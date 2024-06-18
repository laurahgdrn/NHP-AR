ann_file = 'C:/Users/hagedorn/Desktop/data/macaques_skeleton_2d_train2.pkl'
auto_scale_lr = dict(base_batch_size=128, enable=False)
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=False, interval=1, show=False, type='VisualizationHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        graph_cfg=dict(layout='coco', mode='stgcn_spatial'), type='STGCN'),
    cls_head=dict(in_channels=256, num_classes=2, type='GCNHead'),
    type='RecognizerGCN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.1, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/hagedorn/Desktop/data/macaques_skeleton_2d_train2.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'jm',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=10,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=10, test_mode=True,
        type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=5)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        dataset=dict(
            ann_file=
            'C:/Users/hagedorn/Desktop/data/macaques_skeleton_2d_train2.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(dataset='coco', feats=[
                    'jm',
                ], type='GenSkeFeat'),
                dict(clip_len=100, type='UniformSampleFrames'),
                dict(type='PoseDecode'),
                dict(num_person=2, type='FormatGCNInput'),
                dict(type='PackActionInputs'),
            ],
            split='xsub_train',
            type='PoseDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(clip_len=100, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        'C:/Users/hagedorn/Desktop/data/macaques_skeleton_2d_train2.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'jm',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='WandbVisBackend'),
    ])
work_dir = './work_dirs\\stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d'
