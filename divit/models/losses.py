"""
Функции потерь для обучения DI-ViT.

Многокомпонентная функция потерь:
L_total = L_det + λ1 * L_def + λ2 * L_cross + λ3 * L_roof + λ4 * L_connect

где:
λ1 = 0.05 (регуляризация деформаций)
λ2 = 0.5 (согласованность крона-тень)
λ3 = 0.2 (согласованность с геометрией крыши)
λ4 = 0.3 (связывание фрагментов теней)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class FocalLoss(nn.Module):
    """
    Focal Loss для решения проблемы дисбаланса классов.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, N] логиты модели
            targets: [B, N] целевые метки
            
        Returns:
            loss: скалярная потеря
        """
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union Loss для регрессии bounding boxes.
    
    Обеспечивает более точное совпадение предсказанных и истинных рамок.
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: [B, N, 4] или [B, 4, H, W] предсказанные рамки (x, y, w, h)
            target_boxes: [B, N, 4] или [B, 4, H, W] целевые рамки
            
        Returns:
            giou_loss: скалярная потеря
        """
        # Преобразование в формат [B, N, 4] если нужно
        if len(pred_boxes.shape) == 4:
            B, C, H, W = pred_boxes.shape
            # Ожидаем что C = 4 или C = 5 (x, y, w, h, angle)
            # Берём только первые 4 канала для GIoU
            pred_boxes = pred_boxes[:, :4, :, :].permute(0, 2, 3, 1).reshape(B, H * W, 4)
            target_boxes = target_boxes[:, :4, :, :].permute(0, 2, 3, 1).reshape(B, H * W, 4)
        
        # Преобразование в формат (x1, y1, x2, y2)
        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)
        
        # Пересечение
        inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
        inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
        inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
        inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Площаи
        pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
        target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]) * (target_xyxy[..., 3] - target_xyxy[..., 1])
        
        union_area = pred_area + target_area - inter_area
        
        # Охватывающая рамка
        enclose_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
        enclose_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
        enclose_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
        enclose_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + self.eps
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area
        
        loss = 1 - giou
        return loss.mean()
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Преобразование из (x, y, w, h) в (x1, y1, x2, y2)."""
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)


class DeformationSmoothnessLoss(nn.Module):
    """
    Регуляризация плавности деформаций L_def.
    
    Штрафует большие значения градиента поля смещения:
    L_def = (1/N) * Σ_i (||∇Δ_tree(p_i)||_F^2 + ||∇Δ_shadow(p_i)||_F^2)
    
    Использует норму Фробениуса для матрицы Якоби двумерного поля смещения.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, offsets_tree: List[torch.Tensor], 
                offsets_shadow: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            offsets_tree: список [B, N, 2] полей смещений для крон
            offsets_shadow: опционально, список полей смещений для теней
            
        Returns:
            loss: скалярная потеря плавности
        """
        total_loss = 0.0
        count = 0
        
        for offsets in offsets_tree:
            # Вычисление градиента через конечные разности
            grad = self._compute_gradient(offsets)
            # Норма Фробениуса
            frobenius_norm = torch.norm(grad, p='fro', dim=-1)
            total_loss += frobenius_norm.pow(2).mean()
            count += 1
        
        if offsets_shadow is not None:
            for offsets in offsets_shadow:
                grad = self._compute_gradient(offsets)
                frobenius_norm = torch.norm(grad, p='fro', dim=-1)
                total_loss += frobenius_norm.pow(2).mean()
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=offsets_tree[0].device)
    
    def _compute_gradient(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Вычисление градиента поля смещения.
        
        Args:
            offsets: [B, N, 2] поле смещений
            
        Returns:
            grad: [B, N, 2, 2] градиент (Якобиан)
        """
        B, N, _ = offsets.shape
        H = W = int(math.sqrt(N))
        
        # Reshape к пространственной сетке
        offsets_grid = offsets.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Градиенты по x и y
        grad_x = torch.gradient(offsets_grid, dim=3)[0]  # [B, 2, H, W]
        grad_y = torch.gradient(offsets_grid, dim=2)[0]  # [B, 2, H, W]
        
        # Стекирование в Якобиан
        grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, 2, H, W, 2]
        grad = grad.permute(0, 2, 3, 1, 4).reshape(B, H * W, 2, 2)  # [B, N, 2, 2]
        
        return grad


class CrossConsistencyLoss(nn.Module):
    """
    Потеря согласованности «крона–тень» L_cross.
    
    Явно связывает геометрические центры кроны и её тени,
    используя априорную информацию о положении солнца:
    
    L_cross = Σ_{i,j} ||(p_j^shadow - p_i^tree) - d_proj(θ_sun, h_tree)||^2 * I_{(i,j) ∈ pairs}
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, tree_centers: torch.Tensor, shadow_centers: torch.Tensor,
                sun_azimuth: torch.Tensor, tree_heights: torch.Tensor,
                pairs_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tree_centers: [B, N_tree, 2] центры крон
            shadow_centers: [B, N_shadow, 2] центры теней
            sun_azimuth: [B, 2] sin/cos азимута солнца
            tree_heights: [B, N_tree] высоты деревьев
            pairs_mask: [B, N_tree, N_shadow] маска пар крона-тень
            
        Returns:
            loss: скалярная потеря согласованности
        """
        B, N_tree, _ = tree_centers.shape
        N_shadow = shadow_centers.shape[1]
        
        # Вычисление ожидаемого вектора смещения тени
        # d_proj зависит от азимута солнца и высоты дерева
        sin_azimuth = sun_azimuth[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        cos_azimuth = sun_azimuth[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        
        # Проекция тени пропорциональна высоте дерева и углу солнца
        shadow_length = tree_heights.unsqueeze(-1) * torch.abs(cos_azimuth)  # [B, N_tree, 1]
        
        # Ожидаемое смещение
        expected_offset = torch.cat([
            shadow_length * sin_azimuth.expand(-1, N_tree, -1),
            shadow_length * cos_azimuth.expand(-1, N_tree, -1)
        ], dim=-1)  # [B, N_tree, 2]
        
        # Фактическое смещение для каждой пары
        actual_offsets = shadow_centers.unsqueeze(1) - tree_centers.unsqueeze(2)  # [B, N_tree, N_shadow, 2]
        expected_offsets = expected_offset.unsqueeze(2).expand(-1, -1, N_shadow, -1)
        
        # Квадрат евклидова расстояния
        diff = actual_offsets - expected_offsets  # [B, N_tree, N_shadow, 2]
        squared_dist = diff.pow(2).sum(dim=-1)  # [B, N_tree, N_shadow]
        
        # Применение маски пар
        loss = (squared_dist * pairs_mask).sum() / (pairs_mask.sum() + 1e-6)
        
        return loss


class RoofConsistencyLoss(nn.Module):
    """
    Потеря согласованности с геометрией крыши L_roof.
    
    Штрафует предсказания полей смещения, не соответствующие геометрии крыши.
    Наибольшие деформации тени ожидаются вблизи коньков крыш:
    
    L_roof = Σ_{p ∈ ridge} M_shadow(p) * max(0, τ_ridge - ||∇Δ_shadow(p)||_F)
    
    где τ_ridge = 0.5 -- пороговое значение градиента.
    """
    
    def __init__(self, tau_ridge: float = 0.5):
        super().__init__()
        self.tau_ridge = tau_ridge
        
    def forward(self, offsets_shadow: torch.Tensor, ridge_mask: torch.Tensor,
                shadow_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offsets_shadow: [B, H, W, 2] поле смещений теней
            ridge_mask: [B, H, W] бинарная маска коньков
            shadow_mask: [B, H, W] бинарная маска наличия тени
            
        Returns:
            loss: скалярная потеря согласованности с крышей
        """
        # Вычисление градиента
        grad = self._compute_gradient(offsets_shadow)  # [B, H, W, 2, 2]
        
        # Норма Фробениуса
        frobenius_norm = torch.norm(grad, p='fro', dim=-1).mean(dim=-1)  # [B, H, W]
        
        # Hinge-подобная функция
        hinge_loss = torch.clamp(self.tau_ridge - frobenius_norm, min=0)  # [B, H, W]
        
        # Применение масок
        mask = ridge_mask * shadow_mask  # Только коньки с тенью
        loss = (hinge_loss * mask).sum() / (mask.sum() + 1e-6)
        
        return loss
    
    def _compute_gradient(self, offsets: torch.Tensor) -> torch.Tensor:
        """Вычисление градиента поля смещения."""
        B, H, W, _ = offsets.shape
        offsets_grid = offsets.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        grad_x = torch.gradient(offsets_grid, dim=3)[0]
        grad_y = torch.gradient(offsets_grid, dim=2)[0]
        
        grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, 2, H, W, 2]
        grad = grad.permute(0, 2, 3, 1, 4)  # [B, H, W, 2, 2]
        
        return grad


class ShadowConnectionLoss(nn.Module):
    """
    Потеря связывания фрагментов теней L_connect.
    
    Бинарная кросс-энтропия для классификации связности пар фрагментов:
    
    L_connect = -Σ_{(i,j) ∈ frags} [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [N_pairs] предсказанные вероятности связности
            targets: [N_pairs] бинарные метки связности
            
        Returns:
            loss: скалярная потеря связывания
        """
        return self.bce_loss(predictions, targets.float())


class DIViTLoss(nn.Module):
    """
    Полная многокомпонентная функция потерь DI-ViT.
    
    L_total = L_det + λ1 * L_def + λ2 * L_cross + λ3 * L_roof + λ4 * L_connect
    
    Детекционная потеря:
    L_det = L_cls + L_box + L_giou
    """
    
    def __init__(self, lambda_def: float = 0.05, lambda_cross: float = 0.5,
                 lambda_roof: float = 0.2, lambda_connect: float = 0.3):
        super().__init__()
        
        self.lambda_def = lambda_def
        self.lambda_cross = lambda_cross
        self.lambda_roof = lambda_roof
        self.lambda_connect = lambda_connect
        
        # Компоненты детекционной потери
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = nn.L1Loss()
        self.giou_loss = GIoULoss()
        
        # Остальные компоненты
        self.def_loss = DeformationSmoothnessLoss()
        self.cross_loss = CrossConsistencyLoss()
        self.roof_loss = RoofConsistencyLoss(tau_ridge=0.5)
        self.connect_loss = ShadowConnectionLoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: dict с предсказаниями модели:
                - cls_logits: логиты классов
                - box_pred: предсказания рамок
                - offsets_tree: поля смещений для крон
                - offsets_shadow: опционально, поля смещений для теней
                - sun_encoding: оценка азимута солнца
                - roof_types: типы поверхности крыши
                
            targets: dict с целевыми значениями:
                - cls_targets: метки классов
                - box_targets: целевые рамки
                - tree_centers: центры крон
                - shadow_centers: центры теней
                - sun_azimuth: истинный азимут солнца
                - tree_heights: высоты деревьев
                - pairs_mask: маска пар крона-тень
                - ridge_mask: маска коньков
                - shadow_mask: маска теней
                - connection_preds: предсказания связности
                - connection_targets: метки связности
                
        Returns:
            total_loss: общая потеря
            losses: dict с отдельными компонентами потерь
        """
        losses = {}
        
        # Детекционная потеря
        cls_loss = self.focal_loss(outputs['cls_logits'], targets['cls_targets'])
        box_loss = self.box_loss(outputs['box_pred'], targets['box_targets'])
        giou_loss = self.giou_loss(outputs['box_pred'], targets['box_targets'])
        
        det_loss = cls_loss + box_loss + giou_loss
        losses['det'] = det_loss
        losses['cls'] = cls_loss
        losses['box'] = box_loss
        losses['giou'] = giou_loss
        
        # Регуляризация деформаций
        if outputs.get('offsets_tree') is not None:
            def_loss = self.def_loss(
                outputs['offsets_tree'],
                outputs.get('offsets_shadow')
            )
            losses['def'] = def_loss
        else:
            losses['def'] = torch.tensor(0.0, device=outputs['cls_logits'].device)
        
        # Согласованность крона-тень
        if 'tree_centers' in targets and 'shadow_centers' in targets:
            cross_loss = self.cross_loss(
                targets['tree_centers'],
                targets['shadow_centers'],
                targets['sun_azimuth'],
                targets['tree_heights'],
                targets['pairs_mask']
            )
            losses['cross'] = cross_loss
        else:
            losses['cross'] = torch.tensor(0.0, device=outputs['cls_logits'].device)
        
        # Согласованность с геометрией крыши
        if outputs.get('offsets_shadow') is not None and 'ridge_mask' in targets:
            roof_loss = self.roof_loss(
                outputs['offsets_shadow'],
                targets['ridge_mask'],
                targets.get('shadow_mask', torch.ones_like(targets['ridge_mask']))
            )
            losses['roof'] = roof_loss
        else:
            losses['roof'] = torch.tensor(0.0, device=outputs['cls_logits'].device)
        
        # Связывание фрагментов теней
        if 'connection_preds' in outputs and 'connection_targets' in targets:
            connect_loss = self.connect_loss(
                outputs['connection_preds'],
                targets['connection_targets']
            )
            losses['connect'] = connect_loss
        else:
            losses['connect'] = torch.tensor(0.0, device=outputs['cls_logits'].device)
        
        # Общая потеря
        total_loss = (
            losses['det'] +
            self.lambda_def * losses['def'] +
            self.lambda_cross * losses['cross'] +
            self.lambda_roof * losses['roof'] +
            self.lambda_connect * losses['connect']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


if __name__ == '__main__':
    # Тест функции потерь
    loss_fn = DIViTLoss()
    
    # Пример выходов модели
    outputs = {
        'cls_logits': torch.randn(2, 3, 32, 32),
        'box_pred': torch.randn(2, 5, 32, 32),
        'offsets_tree': [torch.randn(2, 1024, 2)],
        'sun_encoding': torch.randn(2, 2),
    }
    
    # Пример целевых значений
    targets = {
        'cls_targets': torch.randint(0, 2, (2, 3, 32, 32)).float(),
        'box_targets': torch.randn(2, 5, 32, 32),
        'tree_centers': torch.randn(2, 10, 2),
        'shadow_centers': torch.randn(2, 10, 2),
        'sun_azimuth': torch.randn(2, 2),
        'tree_heights': torch.randn(2, 10),
        'pairs_mask': torch.randint(0, 2, (2, 10, 10)).float(),
        'ridge_mask': torch.randint(0, 2, (2, 32, 32)).float(),
        'shadow_mask': torch.randint(0, 2, (2, 32, 32)).float(),
        'connection_preds': torch.rand(20),
        'connection_targets': torch.randint(0, 2, (20,)).float(),
    }
    
    total_loss, losses = loss_fn(outputs, targets)
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
