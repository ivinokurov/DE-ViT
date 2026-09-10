"""
Инициализация пакета моделей DE-ViT.
"""

from .devit import (
    DEViT,
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
    DEViTLoss,
    FocalLoss,
    GIoULoss,
    DeformationSmoothnessLoss,
    CrossConsistencyLoss,
    RoofConsistencyLoss,
    ShadowConnectionLoss,
)

__all__ = [
    # Модели
    'DEViT',
    'HierarchicalBackbone',
    'DeformableAttention',
    'DualDeformableAttention',
    'CrossAttentionWithSun',
    'RoofGeometryModule',
    'DetectionHead',
    'SunAzimuthHead',
    'ShadowConnectionHead',
    
    # Функции потерь
    'DEViTLoss',
    'FocalLoss',
    'GIoULoss',
    'DeformationSmoothnessLoss',
    'CrossConsistencyLoss',
    'RoofConsistencyLoss',
    'ShadowConnectionLoss',
]