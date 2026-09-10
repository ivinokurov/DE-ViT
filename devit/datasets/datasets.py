"""
Датасеты для обучения и оценки DE-ViT.

Поддерживаемые датасеты:
- OAM-TCD: 5,072 изображения с >280,000 размеченных крон деревьев
- ISPRS Vaihingen: 16 изображений высокого разрешения с разметкой зданий
- Ciyutuo Village: 65 изображений с векторной разметкой зданий и теней
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2


class OAHTCDDataset(Dataset):
    """
    Датасет OAM-TCD для предобучения детекции крон.
    
    5,072 изображения с более чем 280,000 размеченных крон деревьев.
    Используется для предобучения backbone детекции крон.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 1024,
                 transform=None, augment: bool = False):
        """
        Args:
            root_dir: корневая директория датасета
            split: 'train', 'val' или 'test'
            img_size: размер входных изображений
            transform: дополнительные трансформации
            augment: применять ли аугментации
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.augment = augment
        
        # Загрузка метаданных
        self.annotations = self._load_annotations()
        
        # Нормализация ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def _load_annotations(self) -> List[Dict]:
        """Загрузка аннотаций из JSON файла."""
        ann_file = self.root_dir / f'annotations_{self.split}.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for item in data:
            annotations.append({
                'image_path': self.root_dir / 'images' / item['image_name'],
                'trees': item.get('trees', []),  # Список bounding boxes крон
                'image_id': item.get('image_id', '')
            })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        
        # Загрузка изображения
        image = Image.open(ann['image_path']).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Аугментации
        if self.augment:
            image = self._apply_augmentations(image)
        
        # Преобразование в тензор
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # Подготовка целевых значений
        targets = self._prepare_targets(ann, image.size)
        
        return {
            'image': image_tensor,
            'targets': targets,
            'image_id': ann['image_id'],
            'image_path': str(ann['image_path'])
        }
    
    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Применение аугментаций: повороты, масштабирование, цветовой джиттер."""
        # Случайный поворот
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR)
        
        # Случайное масштабирование
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.BILINEAR)
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Цветовой джиттер (имитация изменений освещения)
        if np.random.rand() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))
        
        return image
    
    def _prepare_targets(self, ann: Dict, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Подготовка целевых значений для обучения."""
        trees = ann['trees']
        
        if len(trees) == 0:
            return {
                'cls_targets': torch.zeros(1, dtype=torch.long),
                'box_targets': torch.zeros(1, 5),
                'tree_centers': torch.zeros(1, 2),
                'tree_heights': torch.zeros(1),
            }
        
        # Извлечение bounding boxes и классов
        boxes = []
        classes = []
        centers = []
        heights = []
        
        for tree in trees:
            x, y, w, h = tree['bbox']  # [x, y, width, height]
            
            # Нормализация координат
            x_norm = x / image_size[0]
            y_norm = y / image_size[1]
            w_norm = w / image_size[0]
            h_norm = h / image_size[1]
            
            boxes.append([x_norm, y_norm, w_norm, h_norm, tree.get('angle', 0)])
            classes.append(tree.get('class_id', 0))  # 0 для крон
            centers.append([x_norm + w_norm/2, y_norm + h_norm/2])
            heights.append(tree.get('height', 1.0))  # Высота дерева
        
        return {
            'cls_targets': torch.tensor(classes, dtype=torch.long),
            'box_targets': torch.tensor(boxes, dtype=torch.float32),
            'tree_centers': torch.tensor(centers, dtype=torch.float32),
            'tree_heights': torch.tensor(heights, dtype=torch.float32),
        }


class ISPRSVaihingenDataset(Dataset):
    """
    Датасет ISPRS Vaihingen для обучения сегментации крыш.
    
    16 изображений высокого разрешения с детальной разметкой зданий и типов поверхностей.
    Используется для обучения сегментации крыш и оценки устойчивости к теневым артефактам.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 1024,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Типы поверхностей: 0-грунт, 1-плоская крыша, 2-скат, 3-конёк
        self.num_classes = 4
        
        self.image_paths, self.mask_paths = self._load_paths()
        
    def _load_paths(self) -> Tuple[List[Path], List[Path]]:
        """Загрузка путей к изображениям и маскам."""
        image_dir = self.root_dir / 'images'
        mask_dir = self.root_dir / 'masks'
        
        image_paths = sorted(list(image_dir.glob('*.tif'))) + sorted(list(image_dir.glob('*.png')))
        mask_paths = [mask_dir / p.name.replace('.tif', '.png').replace('.jpg', '.png') 
                      for p in image_paths]
        
        # Фильтрация несуществующих файлов
        valid_pairs = [(img, mask) for img, mask in zip(image_paths, mask_paths) 
                       if img.exists() and mask.exists()]
        
        image_paths, mask_paths = zip(*valid_pairs) if valid_pairs else ([], [])
        return list(image_paths), list(mask_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Загрузка изображения
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Загрузка маски
        mask = Image.open(self.mask_paths[idx])
        mask = mask.resize((self.img_size // 4, self.img_size // 4), Image.NEAREST)
        
        # Преобразование изображения
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # Преобразование маски
        mask_np = np.array(mask).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np)
        
        # Создание масок для разных типов поверхностей
        roof_masks = {}
        for i in range(self.num_classes):
            roof_masks[f'roof_type_{i}'] = (mask_tensor == i).float()
        
        return {
            'image': image_tensor,
            'roof_mask': mask_tensor,
            'roof_masks': roof_masks,
            'image_path': str(self.image_paths[idx])
        }


class CiyutuoVillageDataset(Dataset):
    """
    Датасет Ciyutuo Village для целевой настройки и тестирования.
    
    65 изображений с векторной разметкой зданий коттеджного типа.
    Для 20% изображений вручную создана разметка теней.
    Используется для финальной донастройки и тестирования.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 1024,
                 transform=None, include_shadows: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.include_shadows = include_shadows
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Загрузка аннотаций."""
        ann_file = self.root_dir / f'annotations_{self.split}.json'
        if not ann_file.exists():
            # Если файл аннотаций не найден, создаём список изображений
            image_dir = self.root_dir / 'images'
            image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
            
            annotations = []
            for img_path in image_paths:
                annotations.append({
                    'image_path': img_path,
                    'trees': [],
                    'shadows': [],
                    'buildings': [],
                    'sun_azimuth': 0.0,
                    'image_id': img_path.stem
                })
            return annotations
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for item in data:
            annotations.append({
                'image_path': self.root_dir / 'images' / item['image_name'],
                'trees': item.get('trees', []),
                'shadows': item.get('shadows', []),
                'buildings': item.get('buildings', []),
                'sun_azimuth': item.get('sun_azimuth', 0.0),
                'ridge_mask_path': item.get('ridge_mask'),
                'shadow_fragments': item.get('shadow_fragments', []),
                'fragment_connections': item.get('fragment_connections', []),
                'image_id': item.get('image_id', '')
            })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        
        # Загрузка изображения
        image = Image.open(ann['image_path']).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Преобразование
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # Кодирование азимута солнца
        sun_azimuth = ann.get('sun_azimuth', 0.0)
        sun_encoding = torch.tensor([np.sin(sun_azimuth), np.cos(sun_azimuth)], dtype=torch.float32)
        
        # Подготовка целевых значений
        targets = self._prepare_targets(ann, image.size)
        targets['sun_azimuth'] = sun_encoding
        
        return {
            'image': image_tensor,
            'targets': targets,
            'image_id': ann['image_id'],
            'sun_azimuth': sun_encoding
        }
    
    def _prepare_targets(self, ann: Dict, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Подготовка всех целевых значений."""
        targets = {}
        
        # Обработка крон
        trees = ann.get('trees', [])
        if trees:
            tree_boxes, tree_classes, tree_centers, tree_heights = self._process_objects(
                trees, image_size, class_offset=0
            )
            targets['tree_boxes'] = tree_boxes
            targets['tree_centers'] = tree_centers
            targets['tree_heights'] = tree_heights
        else:
            targets['tree_boxes'] = torch.zeros(0, 5)
            targets['tree_centers'] = torch.zeros(0, 2)
            targets['tree_heights'] = torch.zeros(0)
        
        # Обработка теней
        shadows = ann.get('shadows', [])
        if shadows and self.include_shadows:
            shadow_boxes, shadow_classes, shadow_centers, _ = self._process_objects(
                shadows, image_size, class_offset=1
            )
            targets['shadow_boxes'] = shadow_boxes
            targets['shadow_centers'] = shadow_centers
        else:
            targets['shadow_boxes'] = torch.zeros(0, 5)
            targets['shadow_centers'] = torch.zeros(0, 2)
        
        # Объединение для детекции
        all_boxes = torch.cat([targets['tree_boxes'], targets['shadow_boxes']], dim=0)
        all_classes = torch.cat([
            torch.zeros(len(targets['tree_boxes'])),
            torch.ones(len(targets['shadow_boxes']))
        ], dim=0) if len(all_boxes) > 0 else torch.zeros(0)
        
        targets['box_targets'] = all_boxes
        targets['cls_targets'] = all_classes.long()
        
        # Маска пар крона-тень
        n_trees = len(targets['tree_boxes'])
        n_shadows = len(targets['shadow_boxes'])
        if n_trees > 0 and n_shadows > 0:
            pairs_mask = self._create_pairs_mask(
                targets['tree_centers'], targets['shadow_centers'], 
                ann.get('sun_azimuth', 0.0)
            )
            targets['pairs_mask'] = pairs_mask
        else:
            targets['pairs_mask'] = torch.zeros(max(n_trees, 1), max(n_shadows, 1))
        
        # Загрузка маски коньков
        ridge_mask_path = ann.get('ridge_mask_path')
        if ridge_mask_path and os.path.exists(ridge_mask_path):
            ridge_mask = Image.open(ridge_mask_path)
            ridge_mask = ridge_mask.resize((self.img_size // 4, self.img_size // 4), Image.NEAREST)
            targets['ridge_mask'] = torch.from_numpy(np.array(ridge_mask)).float() / 255.0
        else:
            targets['ridge_mask'] = torch.zeros(self.img_size // 4, self.img_size // 4)
        
        # Связность фрагментов теней
        fragment_connections = ann.get('fragment_connections', [])
        if fragment_connections:
            conn_preds = torch.tensor([c['probability'] for c in fragment_connections])
            conn_targets = torch.tensor([c['label'] for c in fragment_connections]).float()
            targets['connection_preds'] = conn_preds
            targets['connection_targets'] = conn_targets
        else:
            targets['connection_preds'] = torch.zeros(0)
            targets['connection_targets'] = torch.zeros(0)
        
        return targets
    
    def _process_objects(self, objects: List[Dict], image_size: Tuple[int, int],
                         class_offset: int = 0) -> Tuple[torch.Tensor, ...]:
        """Обработка объектов (кроны или тени)."""
        boxes = []
        classes = []
        centers = []
        heights = []
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            
            # Нормализация
            x_norm = x / image_size[0]
            y_norm = y / image_size[1]
            w_norm = w / image_size[0]
            h_norm = h / image_size[1]
            
            boxes.append([x_norm, y_norm, w_norm, h_norm, obj.get('angle', 0)])
            classes.append(obj.get('class_id', 0) + class_offset)
            centers.append([x_norm + w_norm/2, y_norm + h_norm/2])
            heights.append(obj.get('height', 1.0))
        
        return (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5),
            torch.tensor(classes, dtype=torch.long) if classes else torch.zeros(0),
            torch.tensor(centers, dtype=torch.float32) if centers else torch.zeros(0, 2),
            torch.tensor(heights, dtype=torch.float32) if heights else torch.zeros(0)
        )
    
    def _create_pairs_mask(self, tree_centers: torch.Tensor, shadow_centers: torch.Tensor,
                           sun_azimuth: float, distance_threshold: float = 0.3) -> torch.Tensor:
        """Создание маски пар крона-тень на основе геометрии."""
        n_trees = len(tree_centers)
        n_shadows = len(shadow_centers)
        
        if n_trees == 0 or n_shadows == 0:
            return torch.zeros(max(n_trees, 1), max(n_shadows, 1))
        
        pairs_mask = torch.zeros(n_trees, n_shadows)
        
        # Ожидаемое направление тени
        sin_az = np.sin(sun_azimuth)
        cos_az = np.cos(sun_azimuth)
        
        for i in range(n_trees):
            for j in range(n_shadows):
                # Вектор от кроны к тени
                offset = shadow_centers[j] - tree_centers[i]
                offset_norm = torch.norm(offset)
                
                if offset_norm < distance_threshold:
                    # Проверка соответствия направления
                    expected_dir = torch.tensor([sin_az, cos_az])
                    actual_dir = offset / (offset_norm + 1e-6)
                    
                    dir_similarity = torch.dot(expected_dir, actual_dir)
                    if dir_similarity > 0.5:  # Порог схожести направлений
                        pairs_mask[i, j] = 1.0
        
        return pairs_mask


def create_dataloader(dataset: Dataset, batch_size: int = 4, num_workers: int = 4,
                      shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    """Создание DataLoader с оптимальными параметрами."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Функция для батчирования данных."""
    images = torch.stack([item['image'] for item in batch])
    
    result = {'images': images}
    
    # Добавление дополнительных полей
    for key in batch[0].keys():
        if key == 'image':
            continue
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], dict):
            result[key] = {}
            for subkey in batch[0][key].keys():
                if isinstance(batch[0][key][subkey], torch.Tensor):
                    result[key][subkey] = torch.stack([item[key][subkey] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    
    return result


if __name__ == '__main__':
    # Тест датасетов
    print("Testing datasets...")
    
    # OAM-TCD (пример структуры)
    # oam_dataset = OAHTCDDataset(root_dir='/path/to/oam-tcd', split='train', augment=True)
    # print(f"OAM-TCD: {len(oam_dataset)} images")
    
    # ISPRS Vaihingen
    # isprs_dataset = ISPRSVaihingenDataset(root_dir='/path/to/isprs', split='train')
    # print(f"ISPRS Vaihingen: {len(isprs_dataset)} images")
    
    # Ciyutuo Village
    # ciyutuo_dataset = CiyutuoVillageDataset(root_dir='/path/to/ciyutuo', split='train')
    # print(f"Ciyutuo Village: {len(ciyutuo_dataset)} images")
    
    print("Dataset classes created successfully!")
