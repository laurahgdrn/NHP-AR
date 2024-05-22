import torch
from mmaction.registry import MODELS
from mmaction.registry import METRICS
from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)

import mmcv
import torch
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample
import torch.optim as optim
from mmengine import track_iter_progress
import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule, Sequential
from mmengine.structures import LabelData
from mmaction.registry import MODELS
from mmengine.model import BaseDataPreprocessor, stack_batch
import copy
from collections import OrderedDict
from mmengine.evaluator import BaseMetric
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import METRICS
from mmengine.dataset import Compose


import torch
import copy
from mmaction.registry import MODELS

model_cfg = dict(
    type='RecognizerZelda',
    backbone=dict(type='BackBoneZelda'),
    cls_head=dict(
        type='ClsHeadZelda',
        num_classes=2,
        in_channels=128,
        average_clips='prob'),
    data_preprocessor = dict(
        type='DataPreprocessorZelda',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    def __init__(self, topk=(1, 5), collect_device='cpu', prefix='acc'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = topk

    def process(self, data_batch, data_samples):
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            scores = data_sample['pred_score']['item'].cpu().numpy()
            label = data_sample['gt_label'].item()
            result['scores'] = scores
            result['label'] = label
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        eval_results = OrderedDict()
        labels = [res['label'] for res in results]
        scores = [res['scores'] for res in results]
        topk_acc = top_k_accuracy(scores, labels, self.topk)
        for k, acc in zip(self.topk, topk_acc):
            eval_results[f'topk{k}'] = acc
        return eval_results
model = MODELS.build(model_cfg)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare test data
test_dataset = DATASETS.build(val_dataset_cfg)

from mmaction.registry import METRICS

metric_cfg = dict(type='AccuracyMetric', topk=(1, 2))

metric = METRICS.build(metric_cfg)

# data_samples = [d.to_dict() for d in predictions]

# metric.process(batched_packed_results, data_samples)
# acc = metric.compute_metrics(metric.results)

# Run inference
results = []
with torch.no_grad():
    for data_batch in val_data_loader:
        data = model.data_preprocessor(data_batch, training=False)
        predictions = model(**data, mode='predict')
        results.extend(predictions)

# Evaluate performance
metric = METRICS.build(metric_cfg)
data_samples = [d.to_dict() for d in results]
metric.process(data_batch, data_samples)
evaluation_results = metric.compute_metrics(metric.results)
print("Evaluation results:", evaluation_results)

# Visualize results (optional)
# You can visualize predictions, ground truth labels, and any other relevant information to understand the model's performance better.
