"""
Скрипт обучения DI-ViT.

Двухстадийная стратегия обучения:
1. Предобучение backbone на OAM-TCD для детекции крон
2. Донастройка всей архитектуры на комбинированном наборе данных (ISPRS Vaihingen + Ciyutuo Village)

Гиперпараметры:
- Оптимизатор: AdamW
- Начальная скорость обучения: 1e-4
- Планировщик: косинусный
- Количество эпох: 100
- Размер батча: 4
- Размер изображения: 1024x1024
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Импорт компонентов DI-ViT
from models.divit import DIViT
from models.losses import DIViTLoss
from datasets.datasets import (
    OAHTCDDataset, 
    ISPRSVaihingenDataset, 
    CiyutuoVillageDataset,
    create_dataloader,
    collate_fn
)


class Trainer:
    """Тренер для обучения DI-ViT."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Настройка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Инициализация модели
        self.model = DIViT(
            img_size=config['img_size'],
            in_chans=3
        ).to(self.device)
        
        # Инициализация функции потерь
        self.criterion = DIViTLoss(
            lambda_def=config.get('lambda_def', 0.05),
            lambda_cross=config.get('lambda_cross', 0.5),
            lambda_roof=config.get('lambda_roof', 0.2),
            lambda_connect=config.get('lambda_connect', 0.3)
        )
        
        # Оптимизатор
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Планировщик скорости обучения
        self.scheduler = None
        
        # AMP scaler для смешанной точности
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Метрики
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        # Настройка логгирования
        self._setup_logging()
        
    def _setup_logging(self):
        """Настройка логгирования."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'train_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_dataloaders(self, stage: str = 'pretrain'):
        """Подготовка DataLoader для разных стадий обучения."""
        if stage == 'pretrain':
            # Стадия 1: предобучение на OAM-TCD
            train_dataset = OAHTCDDataset(
                root_dir=self.config['oam_tcd_root'],
                split='train',
                img_size=self.config['img_size'],
                augment=True
            )
            val_dataset = OAHTCDDataset(
                root_dir=self.config['oam_tcd_root'],
                split='val',
                img_size=self.config['img_size'],
                augment=False
            )
        elif stage == 'finetune':
            # Стадия 2: донастройка на комбинированном датасете
            # Здесь можно использовать ConcatDataset для объединения нескольких датасетов
            train_dataset = CiyutuoVillageDataset(
                root_dir=self.config['ciyutuo_root'],
                split='train',
                img_size=self.config['img_size'],
                include_shadows=True
            )
            val_dataset = CiyutuoVillageDataset(
                root_dir=self.config['ciyutuo_root'],
                split='val',
                img_size=self.config['img_size'],
                include_shadows=True
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            shuffle=True
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            shuffle=False
        )
        
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Val dataset size: {len(val_dataset)}")
        
    def train_epoch(self, epoch: int) -> float:
        """Обучение за одну эпоху."""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'det': 0.0, 'def': 0.0, 'cross': 0.0, 
            'roof': 0.0, 'connect': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            targets_dict = {}
            
            # Подготовка целевых значений
            if 'targets' in batch:
                targets = batch['targets']
                if isinstance(targets, dict):
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets_dict[k] = v.to(self.device)
            
            # Добавление дополнительных полей из батча
            for key in ['sun_azimuth', 'tree_centers', 'shadow_centers', 
                       'tree_heights', 'pairs_mask', 'ridge_mask', 
                       'connection_preds', 'connection_targets']:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    targets_dict[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass с AMP
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(images)
                
                # Подготовка targets для функции потерь
                if 'box_targets' not in targets_dict and 'box_targets' in targets:
                    targets_dict['box_targets'] = targets['box_targets'].to(self.device)
                if 'cls_targets' not in targets_dict and 'cls_targets' in targets:
                    targets_dict['cls_targets'] = targets['cls_targets'].to(self.device)
                
                # Вычисление потери
                total_loss_batch, loss_dict = self.criterion(outputs, targets_dict)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                self.optimizer.step()
            
            # Обновление метрик
            total_loss += total_loss_batch.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'det': f'{loss_components["det"] / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        
        self.logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
        for key, value in loss_components.items():
            avg_comp = value / len(self.train_loader)
            self.logger.info(f"  {key}: {avg_comp:.4f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Валидация."""
        self.model.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Validation {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            targets_dict = {}
            
            if 'targets' in batch:
                targets = batch['targets']
                if isinstance(targets, dict):
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets_dict[k] = v.to(self.device)
            
            for key in ['sun_azimuth', 'tree_centers', 'shadow_centers',
                       'tree_heights', 'pairs_mask', 'ridge_mask',
                       'connection_preds', 'connection_targets']:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    targets_dict[key] = batch[key].to(self.device)
            
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(images)
                total_loss_batch, _ = self.criterion(outputs, targets_dict)
            
            total_loss += total_loss_batch.item()
            pbar.set_postfix({'val_loss': f'{total_loss / (batch_idx + 1):.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        
        self.logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}")
        
        # Сохранение лучшей модели
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_loss
    
    def train(self, epochs: int, stage: str = 'pretrain'):
        """Полный цикл обучения."""
        self.prepare_dataloaders(stage)
        
        # Инициализация планировщика
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        self.logger.info(f"Starting {stage} training for {epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Обучение
            train_loss = self.train_epoch(epoch)
            
            # Валидация
            val_loss = self.validate(epoch)
            
            # Обновление планировщика
            self.scheduler.step()
            
            # Логгирование времени эпохи
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
            self.logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохранение чекпоинта каждые N эпох
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
        
        # Сохранение истории обучения
        self.save_history()
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохранение чекпоинта модели."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
            'history': self.history
        }
        
        # Сохранение последнего чекпоинта
        torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pth')
        
        # Сохранение лучшего чекпоинта
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pth')
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        # Сохранение чекпоинта по номеру эпохи
        if epoch % self.config.get('save_interval', 10) == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Загрузка чекпоинта."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'metrics': []})
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
    def save_history(self):
        """Сохранение истории обучения."""
        history_path = Path(self.config.get('log_dir', 'logs')) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Основная функция обучения."""
    parser = argparse.ArgumentParser(description='DI-ViT Training')
    
    # Пути к данным
    parser.add_argument('--oam-tcd-root', type=str, required=True,
                        help='Путь к датасету OAM-TCD')
    parser.add_argument('--ciyutuo-root', type=str, required=True,
                        help='Путь к датасету Ciyutuo Village')
    parser.add_argument('--isprs-root', type=str, default=None,
                        help='Путь к датасету ISPRS Vaihingen')
    
    # Параметры обучения
    parser.add_argument('--stage', type=str, default='pretrain',
                        choices=['pretrain', 'finetune'],
                        help='Стадия обучения')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Начальная скорость обучения')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Минимальная скорость обучения')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='Размер входного изображения')
    
    # Гиперпараметры функции потерь
    parser.add_argument('--lambda-def', type=float, default=0.05,
                        help='Вес регуляризации деформаций')
    parser.add_argument('--lambda-cross', type=float, default=0.5,
                        help='Вес согласованности крона-тень')
    parser.add_argument('--lambda-roof', type=float, default=0.2,
                        help='Вес согласованности с крышей')
    parser.add_argument('--lambda-connect', type=float, default=0.3,
                        help='Вес связывания фрагментов')
    
    # Прочее
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Директория для чекпоинтов')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Директория для логов')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Количество workers для DataLoader')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к чекпоинту для продолжения обучения')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Использовать автоматическое смешанное_precision')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = vars(args)
    
    # Инициализация тренера
    trainer = Trainer(config)
    
    # Загрузка чекпоинта если указано
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Запуск обучения
    trainer.train(epochs=args.epochs, stage=args.stage)


if __name__ == '__main__':
    main()
