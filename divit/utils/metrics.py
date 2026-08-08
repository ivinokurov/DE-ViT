"""
Утилиты для DI-ViT.

Включает:
- Метрики оценки качества (mAP, FPR, FNR)
- Bootstrap для доверительных интервалов
- Визуализация результатов
- Обработка данных
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import cv2


# ============================================================================
# Метрики оценки качества
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Вычисление Intersection over Union (IoU) двух bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] или [x, y, w, h]
        box2: [x1, y1, x2, y2] или [x, y, w, h]
        
    Returns:
        iou: значение IoU в диапазоне [0, 1]
    """
    # Преобразование в формат (x1, y1, x2, y2) если нужно
    if len(box1) == 4 and box1[2] > 0 and box1[3] > 0:
        # Проверка формата (x, y, w, h)
        if box1[2] < (box1[0] + 10) or box1[3] < (box1[1] + 10):
            box1 = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
    
    if len(box2) == 4 and box2[2] > 0 and box2[3] > 0:
        if box2[2] < (box2[0] + 10) or box2[3] < (box2[1] + 10):
            box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])
    
    # Пересечение
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Площади
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Вычисление Average Precision (AP).
    
    Args:
        recall: массив значений recall
        precision: массив значений precision
        
    Returns:
        ap: среднее значение precision
    """
    # Добавление границ
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    
    # Вычисление precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Нахождение точек изменения recall
    i = np.where(recall[1:] != recall[:-1])[0]
    
    # Вычисление AP как площадь под кривой
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    
    return ap


def compute_mAP(detections: List[Dict], ground_truth: List[Dict],
                iou_thresholds: np.ndarray = np.arange(0.5, 0.96, 0.05)) -> Dict[str, float]:
    """
    Вычисление mean Average Precision (mAP).
    
    Args:
        detections: список детекций модели
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}, ...]
        ground_truth: список ground truth объектов
            [{'bbox': [x1, y1, x2, y2], 'class_id': int}, ...]
        iou_thresholds: пороги IoU для усреднения
            
    Returns:
        dict с метриками:
        - mAP@0.5: mAP при IoU=0.5
        - mAP@0.5:0.95: усреднённый mAP по порогам
        - ap_per_class: AP для каждого класса
    """
    # Группировка по классам
    classes = set()
    for det in detections:
        classes.add(det['class_id'])
    for gt in ground_truth:
        classes.add(gt['class_id'])
    
    aps = {threshold: [] for threshold in iou_thresholds}
    
    for class_id in classes:
        # Фильтрация по классу
        class_dets = [d for d in detections if d['class_id'] == class_id]
        class_gts = [g for g in ground_truth if g['class_id'] == class_id]
        
        # Сортировка детекций по уверенности
        class_dets = sorted(class_dets, key=lambda x: x['confidence'], reverse=True)
        
        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(class_dets))
            fp = np.zeros(len(class_dets))
            
            gt_matched = [False] * len(class_gts)
            
            for i, det in enumerate(class_dets):
                det_box = np.array(det['bbox'])
                
                # Поиск лучшего matching GT
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(class_gts):
                    if gt_matched[j]:
                        continue
                    
                    gt_box = np.array(gt['bbox'])
                    iou = compute_iou(det_box, gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1
            
            # Вычисление cumulative TP и FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            # Recall и Precision
            recall = cum_tp / (len(class_gts) + 1e-6)
            precision = cum_tp / (cum_tp + cum_fp + 1e-6)
            
            # Вычисление AP
            ap = compute_ap(recall, precision)
            aps[iou_thresh].append(ap)
    
    # Усреднение по классам
    results = {}
    results['mAP@0.5'] = np.mean(aps[0.5]) if aps[0.5] else 0.0
    
    mAP_all = np.mean([np.mean(aps[t]) for t in iou_thresholds if aps[t]])
    results['mAP@0.5:0.95'] = mAP_all
    
    results['ap_per_class'] = {c: np.mean([aps[t][i] for t in iou_thresholds if len(aps[t]) > i]) 
                               for i, c in enumerate(classes)}
    
    return results


def compute_FPR_FNR(detections: List[Dict], ground_truth: List[Dict],
                    iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Вычисление False Positive Rate (FPR) и False Negative Rate (FNR).
    
    Args:
        detections: список детекций модели
        ground_truth: список ground truth объектов
        iou_threshold: порог IoU для matching
        
    Returns:
        fpr, fnr: доли ложных положительных и ложных отрицательных срабатываний
    """
    if len(ground_truth) == 0 and len(detections) == 0:
        return 0.0, 0.0
    
    gt_matched = [False] * len(ground_truth)
    fp_count = 0
    
    for det in detections:
        det_box = np.array(det['bbox'])
        
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            
            gt_box = np.array(gt['bbox'])
            iou = compute_iou(det_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
        else:
            fp_count += 1
    
    tp_count = sum(gt_matched)
    fn_count = len(ground_truth) - tp_count
    
    fpr = fp_count / (len(detections) + 1e-6) if len(detections) > 0 else 0.0
    fnr = fn_count / (len(ground_truth) + 1e-6) if len(ground_truth) > 0 else 0.0
    
    return fpr, fnr


# ============================================================================
# Bootstrap для доверительных интервалов
# ============================================================================

def bootstrap_confidence_interval(metric_func, data: List[Any],
                                   n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95,
                                   **kwargs) -> Dict[str, float]:
    """
    Вычисление доверительного интервала методом bootstrap.
    
    Args:
        metric_func: функция для вычисления метрики
        data: список элементов данных
        n_bootstrap: количество bootstrap выборок
        confidence_level: уровень доверия (0.95 для 95% CI)
        **kwargs: дополнительные аргументы для metric_func
        
    Returns:
        dict с результатами:
        - mean: среднее значение метрики
        - std: стандартное отклонение
        - ci_lower: нижняя граница доверительного интервала
        - ci_upper: верхняя граница доверительного интервала
    """
    n = len(data)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Bootstrap выборка с заменой
        indices = np.random.choice(n, size=n, replace=True)
        sample = [data[i] for i in indices]
        
        try:
            metric = metric_func(sample, **kwargs)
            bootstrap_metrics.append(metric)
        except Exception:
            continue
    
    if len(bootstrap_metrics) == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    mean = np.mean(bootstrap_metrics)
    std = np.std(bootstrap_metrics)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


# ============================================================================
# Визуализация
# ============================================================================

def visualize_detections(image_path: str, detections: List[Dict],
                         ground_truth: Optional[List[Dict]] = None,
                         output_path: Optional[str] = None,
                         class_names: Dict[int, str] = None) -> np.ndarray:
    """
    Визуализация детекций на изображении.
    
    Args:
        image_path: путь к изображению
        detections: детекции модели
        ground_truth: опционально, ground truth объекты
        output_path: опционально, путь для сохранения
        class_names: словарь имен классов
        
    Returns:
        image: изображение с визуализацией
    """
    if class_names is None:
        class_names = {0: 'Tree', 1: 'Shadow'}
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Цвета для разных классов
    colors = {
        0: (0, 255, 0),   # Tree - зелёный
        1: (255, 0, 0),   # Shadow - красный
    }
    
    # Отрисовка ground truth (если есть)
    if ground_truth:
        for gt in ground_truth:
            bbox = gt['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 2)
            label = class_names.get(gt['class_id'], f"Class {gt['class_id']}")
            cv2.putText(image, f"GT: {label}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Отрисовка детекций
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        color = colors.get(det['class_id'], (255, 255, 0))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = class_names.get(det['class_id'], f"Class {det['class_id']}")
        confidence = det.get('confidence', 0)
        text = f"{label}: {confidence:.2f}"
        
        cv2.putText(image, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Сохранение если указан путь
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return image


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray,
                                 title: str = "Precision-Recall Curve",
                                 output_path: Optional[str] = None) -> None:
    """Построение кривой Precision-Recall."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_history(history: Dict[str, List[float]],
                      output_path: Optional[str] = None) -> None:
    """Построение графика истории потерь."""
    plt.figure(figsize=(10, 6))
    
    for key, values in history.items():
        if 'loss' in key.lower():
            plt.plot(values, label=key, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss History', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# Обработка данных
# ============================================================================

def convert_bbox_format(bbox: List[float], from_format: str = 'xywh',
                        to_format: str = 'xyxy') -> List[float]:
    """
    Конвертация формата bounding box.
    
    Args:
        bbox: bounding box [x, y, w, h] или [x1, y1, x2, y2]
        from_format: исходный формат ('xywh' или 'xyxy')
        to_format: целевой формат
        
    Returns:
        bbox: конвертированный bounding box
    """
    if from_format == to_format:
        return bbox
    
    if from_format == 'xywh' and to_format == 'xyxy':
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    
    elif from_format == 'xyxy' and to_format == 'xywh':
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    else:
        raise ValueError(f"Unknown format conversion: {from_format} -> {to_format}")


def filter_detections_by_confidence(detections: List[Dict],
                                     threshold: float = 0.5) -> List[Dict]:
    """Фильтрация детекций по порогу уверенности."""
    return [d for d in detections if d.get('confidence', 0) >= threshold]


def merge_detections_from_stages(stage1_dets: List[Dict],
                                  stage2_dets: List[Dict],
                                  iou_threshold: float = 0.5) -> List[Dict]:
    """
    Слияние детекций из разных стадий обработки.
    
    Args:
        stage1_dets: детекции из первой стадии
        stage2_dets: детекции из второй стадии
        iou_threshold: порог IoU для удаления дубликатов
        
    Returns:
        merged: объединённые детекции
    """
    all_dets = stage1_dets + stage2_dets
    
    # Сортировка по уверенности
    all_dets = sorted(all_dets, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for det in all_dets:
        is_duplicate = False
        
        for kept_det in keep:
            if det['class_id'] != kept_det['class_id']:
                continue
            
            iou = compute_iou(np.array(det['bbox']), np.array(kept_det['bbox']))
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(det)
    
    return keep


def save_results(results: Dict, output_path: str) -> None:
    """Сохранение результатов в JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def load_results(input_path: str) -> Dict:
    """Загрузка результатов из JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Тест утилит
    print("Testing utilities...")
    
    # Тест IoU
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou(np.array(box1), np.array(box2))
    print(f"IoU test: {iou:.4f}")
    
    # Тест конвертации bbox
    bbox_xywh = [10, 20, 30, 40]
    bbox_xyxy = convert_bbox_format(bbox_xywh, 'xywh', 'xyxy')
    print(f"BBox conversion: {bbox_xywh} -> {bbox_xyxy}")
    
    print("Utilities test completed!")
