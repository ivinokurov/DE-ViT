"""
Инициализация пакета моделей DI-ViT.
"""

from .divit import (
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

from .losses import (
    DIViTLoss,
    FocalLoss,
    GIoULoss,
    DeformationSmoothnessLoss,
    CrossConsistencyLoss,
    RoofConsistencyLoss,
    ShadowConnectionLoss,
)

__all__ = [
    # Модели
    'DIViT',
    'HierarchicalBackbone',
    'DeformableAttention',
    'DualDeformableAttention',
    'CrossAttentionWithSun',
    'RoofGeometryModule',
    'DetectionHead',
    'SunAzimuthHead',
    'ShadowConnectionHead',
    
    # Функции потерь
    'DIViTLoss',
    'FocalLoss',
    'GIoULoss',
    'DeformationSmoothnessLoss',
    'CrossConsistencyLoss',
    'RoofConsistencyLoss',
    'ShadowConnectionLoss',
]