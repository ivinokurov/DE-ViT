"""
Скрипт инференса DI-ViT.

Использование обученной модели для предсказания:
- Детекция крон и теней
- Оценка солнечного азимута
- Сегментация типов поверхности крыши
- Предсказание связности фрагментов теней
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Импорт компонентов DI-ViT
from models.divit import DIViT


class DIViTInference:
    """Класс для инференса DI-ViT."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda',
                 img_size: int = 1024, confidence_threshold: float = 0.5):
        """
        Args:
            checkpoint_path: путь к чекпоинту модели
            device: устройство для вычислений ('cuda' или 'cpu')
            img_size: размер входного изображения
            confidence_threshold: порог уверенности для детекции
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        
        # Загрузка модели
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Нормализация ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Настройка логгирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self, checkpoint_path: str) -> DIViT:
        """Загрузка модели из чекпоинта."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model = DIViT(img_size=self.img_size, in_chans=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        self.logger.info(f"Loaded model from {checkpoint_path}")
        self.logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Предобработка изображения.
        
        Args:
            image_path: путь к изображению
            
        Returns:
            tensor: нормализованный тензор изображения
            meta: метаданные (оригинальный размер и т.д.)
        """
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Ресайз
        image_resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Преобразование в numpy array
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # Нормализация
        image_np = (image_np - self.mean) / self.std
        
        # Преобразование в тензор
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)  # Добавление batch dimension
        
        meta = {
            'original_size': original_size,
            'image_path': image_path,
            'scale_x': original_size[0] / self.img_size,
            'scale_y': original_size[1] / self.img_size
        }
        
        return image_tensor, meta
    
    @torch.no_grad()
    def predict(self, image_path: str, 
                estimate_sun_azimuth: bool = True) -> Dict[str, Any]:
        """
        Предсказание для одного изображения.
        
        Args:
            image_path: путь к изображению
            estimate_sun_azimuth: оценивать ли азимут солнца
            
        Returns:
            dict с результатами:
            - detections: список детекций (кроны и тени)
            - sun_azimuth: оценённый азимут солнца (если requested)
            - roof_types: карта типов поверхности крыши
            - offsets_tree: поле смещений для крон
        """
        # Предобработка
        image_tensor, meta = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor)
        
        # Постобработка результатов
        results = {}
        
        # Детекция объектов
        detections = self._process_detections(
            outputs['cls_logits'],
            outputs['box_pred'],
            meta
        )
        results['detections'] = detections
        
        # Азимут солнца
        if estimate_sun_azimuth:
            sun_encoding = outputs['sun_encoding'].squeeze(0).cpu().numpy()
            sun_azimuth = np.arctan2(sun_encoding[0], sun_encoding[1])
            results['sun_azimuth'] = float(sun_azimuth)
            results['sun_encoding'] = sun_encoding.tolist()
        
        # Типы поверхности крыши
        roof_types = outputs['roof_types'].squeeze(0).cpu().numpy()
        roof_types_map = np.argmax(roof_types, axis=0)
        results['roof_types'] = roof_types_map.astype(int).tolist()
        
        # Поля смещений (если доступны)
        if outputs.get('offsets_tree') is not None:
            results['offsets_tree'] = [
                off.squeeze(0).cpu().numpy().tolist()
                for off in outputs['offsets_tree']
            ]
        
        # Метаданные
        results['meta'] = meta
        
        return results
    
    def _process_detections(self, cls_logits: torch.Tensor, 
                           box_pred: torch.Tensor,
                           meta: Dict) -> List[Dict]:
        """
        Постобработка детекций.
        
        Args:
            cls_logits: [B, num_classes+1, H, W] логиты классов
            box_pred: [B, 5, H, W] предсказания рамок
            meta: метаданные
            
        Returns:
            Список детекций с полями:
            - class: класс объекта ('tree' или 'shadow')
            - confidence: уверенность
            - bbox: bounding box [x, y, w, h]
            - center: центр [cx, cy]
        """
        cls_logits = cls_logits.squeeze(0)  # [num_classes+1, H, W]
        box_pred = box_pred.squeeze(0)  # [5, H, W]
        
        # Применение softmax для получения вероятностей
        cls_probs = F.softmax(cls_logits, dim=0)  # [num_classes+1, H, W]
        
        detections = []
        num_classes = cls_probs.shape[0] - 1  # минус фон
        
        # Поиск пиков уверенности
        for c in range(num_classes):
            class_probs = cls_probs[c + 1]  # Пропускаем фон (класс 0)
            
            # Порог уверенности
            mask = class_probs > self.confidence_threshold
            
            if not mask.any():
                continue
            
            # Нахождение позиций выше порога
            positions = torch.where(mask)
            
            for idx in range(len(positions[0])):
                y, x = positions[0][idx].item(), positions[1][idx].item()
                confidence = class_probs[y, x].item()
                
                # Извлечение bounding box
                box = box_pred[:, y, x]  # [5]
                
                # Преобразование координат
                cx_norm = box[0].item()
                cy_norm = box[1].item()
                w_norm = box[2].item()
                h_norm = box[3].item()
                angle = box[4].item() if len(box) > 4 else 0
                
                # Конвертация в пиксели оригинального изображения
                cx = cx_norm * meta['original_size'][0]
                cy = cy_norm * meta['original_size'][1]
                w = w_norm * meta['original_size'][0]
                h = h_norm * meta['original_size'][1]
                
                bbox = [
                    max(0, cx - w/2),
                    max(0, cy - h/2),
                    w,
                    h
                ]
                
                detections.append({
                    'class': 'tree' if c == 0 else 'shadow',
                    'class_id': c,
                    'confidence': confidence,
                    'bbox': bbox,
                    'center': [cx, cy],
                    'angle': angle,
                    'grid_position': [x, y]
                })
        
        # NMS (Non-Maximum Suppression) для удаления дубликатов
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict], 
                   iou_threshold: float = 0.5) -> List[Dict]:
        """
        Non-Maximum Suppression для удаления перекрывающихся детекций.
        
        Args:
            detections: список детекций
            iou_threshold: порог IoU для объединения
            
        Returns:
            Отфильтрованный список детекций
        """
        if len(detections) == 0:
            return detections
        
        # Сортировка по уверенности
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            # Берём детекцию с максимальной уверенностью
            current = detections.pop(0)
            keep.append(current)
            
            # Удаляем перекрывающиеся детекции того же класса
            remaining = []
            for det in detections:
                if det['class_id'] != current['class_id']:
                    remaining.append(det)
                    continue
                
                iou = self._compute_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Вычисление IoU двух bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Преобразование в формат (x1, y1, x2, y2)
        box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
        
        # Пересечение
        inter_x1 = max(box1_xyxy[0], box2_xyxy[0])
        inter_y1 = max(box1_xyxy[1], box2_xyxy[1])
        inter_x2 = min(box1_xyxy[2], box2_xyxy[2])
        inter_y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Площади
        area1 = w1 * h1
        area2 = w2 * h2
        
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def predict_batch(self, image_paths: List[str], 
                      output_dir: Optional[str] = None) -> List[Dict]:
        """
        Пакетное предсказание для нескольких изображений.
        
        Args:
            image_paths: список путей к изображениям
            output_dir: опционально, директория для сохранения результатов
            
        Returns:
            Список результатов для каждого изображения
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Inference"):
            result = self.predict(image_path)
            results.append(result)
            
            # Сохранение результатов если указана директория
            if output_dir:
                output_path = Path(output_dir) / f"{Path(image_path).stem}_result.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return results
    
    def visualize_results(self, image_path: str, results: Dict, 
                         output_path: str) -> str:
        """
        Визуализация результатов предсказания.
        
        Args:
            image_path: путь к исходному изображению
            results: результаты предсказания
            output_path: путь для сохранения визуализации
            
        Returns:
            output_path: путь к сохранённому изображению
        """
        # Загрузка изображения
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Отрисовка детекций
        for det in results['detections']:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Цвет в зависимости от класса
            color = (0, 255, 0) if det['class'] == 'tree' else (255, 0, 0)
            
            # Рамка
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Подпись
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Сохранение
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        self.logger.info(f"Saved visualization to {output_path}")
        return output_path


def main():
    """Основная функция инференса."""
    parser = argparse.ArgumentParser(description='DI-ViT Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту модели')
    parser.add_argument('--image', type=str, default=None,
                        help='Путь к изображению для предсказания')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Директория с изображениями для пакетного инференса')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для вычислений')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='Размер входного изображения')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Порог уверенности для детекции')
    parser.add_argument('--visualize', action='store_true',
                        help='Сохранять визуализацию результатов')
    
    args = parser.parse_args()
    
    # Инициализация инференса
    inferencer = DIViTInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size,
        confidence_threshold=args.confidence_threshold
    )
    
    # Создание директории для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Обработка одного изображения или директории
    if args.image:
        # Одно изображение
        results = inferencer.predict(args.image)
        
        # Сохранение результатов
        output_path = output_dir / f"{Path(args.image).stem}_result.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        print(f"Detected objects: {len(results['detections'])}")
        
        # Визуализация
        if args.visualize:
            vis_path = output_dir / f"{Path(args.image).stem}_vis.png"
            inferencer.visualize_results(args.image, results, str(vis_path))
    
    elif args.image_dir:
        # Директория с изображениями
        image_paths = list(Path(args.image_dir).glob('*.jpg')) + \
                      list(Path(args.image_dir).glob('*.png'))
        
        print(f"Found {len(image_paths)} images")
        
        results = inferencer.predict_batch(
            [str(p) for p in image_paths],
            output_dir=str(output_dir)
        )
        
        # Визуализация
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for image_path, result in zip(image_paths, results):
                vis_path = vis_dir / f"{image_path.stem}_vis.png"
                inferencer.visualize_results(str(image_path), result, str(vis_path))
        
        print(f"Results saved to {output_dir}")
    
    else:
        print("Please specify --image or --image-dir")


if __name__ == '__main__':
    main()
