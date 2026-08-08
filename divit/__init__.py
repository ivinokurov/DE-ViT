"""
DI-ViT: Deformation-Invariant Visual Transformer

Пакет для реализации деформационно-инвариантного визуального трансформера
для детекции крон деревьев и их теней на аэрофотоснимках.

Архитектура включает:
- Иерархический backbone с деформируемым вниманием
- Раздельные поля деформации для крон (Δ_tree) и теней (Δ_shadow)
- Перекрёстное внимание с учётом азимута солнца
- Модуль геометрии крыши для сегментации типов поверхности
- Многокомпонентную функцию потерь

Автор: DI-ViT Team
"""

from .models.divit import (
    DIViT,
    HierarchicalBackbone,
    DeformableAttention,
    DualDeformableAttention,
    CrossAttentionWithSun,
    RoofGeometryModule,
    DetectionHead,
    SunAzimuthHead,
    ShadowConnectionHead,
)

from .models.losses import (
    DIViTLoss,
    FocalLoss,
    GIoULoss,
    DeformationSmoothnessLoss,
    CrossConsistencyLoss,
    RoofConsistencyLoss,
    ShadowConnectionLoss,
)

from .datasets.datasets import (
    OAHTCDDataset,
    ISPRSVaihingenDataset,
    CiyutuoVillageDataset,
    create_dataloader,
    collate_fn,
)

from .utils.metrics import (
    compute_iou,
    compute_mAP,
    compute_FPR_FNR,
    bootstrap_confidence_interval,
    visualize_detections,
)

__version__ = '1.0.0'
__author__ = 'DI-ViT Team'

__all__ = [
    # Models
    'DIViT',
    'HierarchicalBackbone',
    'DeformableAttention',
    'DualDeformableAttention',
    'CrossAttentionWithSun',
    'RoofGeometryModule',
    'DetectionHead',
    'SunAzimuthHead',
    'ShadowConnectionHead',
    
    # Losses
    'DIViTLoss',
    'FocalLoss',
    'GIoULoss',
    'DeformationSmoothnessLoss',
    'CrossConsistencyLoss',
    'RoofConsistencyLoss',
    'ShadowConnectionLoss',
    
    # Datasets
    'OAHTCDDataset',
    'ISPRSVaihingenDataset',
    'CiyutuoVillageDataset',
    'create_dataloader',
    'collate_fn',
    
    # Metrics
    'compute_iou',
    'compute_mAP',
    'compute_FPR_FNR',
    'bootstrap_confidence_interval',
    'visualize_detections',
]
