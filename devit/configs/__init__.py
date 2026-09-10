"""
Конфигурация DE-ViT.
"""

# Параметры модели
MODEL_CONFIG = {
    'img_size': 1024,
    'in_chans': 3,
    'num_classes': 2,  # кроны, тени
    'num_roof_types': 4,  # грунт, плоская крыша, скат, конёк
    'embed_dims': [64, 128, 256, 512],
    'num_heads': [2, 4, 8, 16],
    'num_blocks': [2, 2, 4, 2],
    'deform_scales': {
        'tree': 0.15,
        'shadow': 0.45,
    },
}

# Гиперпараметры функции потерь
LOSS_WEIGHTS = {
    'lambda_det': 1.0,
    'lambda_def': 0.05,
    'lambda_cross': 0.5,
    'lambda_roof': 0.2,
    'lambda_connect': 0.3,
}

# Параметры обучения
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 8,
    'lr': 1e-4,
    'weight_decay': 0.05,
    'optimizer': 'AdamW',
    'scheduler': 'cosine',
    'img_norm_mean': (0.485, 0.456, 0.406),
    'img_norm_std': (0.229, 0.224, 0.225),
}

# Датасеты
DATASET_PATHS = {
    'oam_tcd': './data/oam_tcd',
    'isprs_vaihingen': './data/isprs_vaihingen',
    'ciyutuo_village': './data/ciyutuo_village',
}