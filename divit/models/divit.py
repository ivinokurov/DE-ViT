"""
Деформационно-инвариантный визуальный трансформер (DI-ViT).

Основная архитектура модели с иерархическим backbone, деформируемым вниманием
и раздельными полями смещения для крон деревьев и их теней.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class DeformableAttention(nn.Module):
    """
    Деформируемое внимание с предсказанием поля смещения.
    
    Для каждого запроса q_i предсказывается масштабируемое поле смещения:
    Δ_ij = s * tanh(W_offset * q_i + b_offset)
    
    Деформированная позиция используется для билинейного семплирования признаков.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, deform_scale: float = 0.15,
                 attention_type: str = 'local'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.deform_scale = deform_scale
        self.attention_type = attention_type  # 'local', 'full', 'global'
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Параметры для предсказания смещений
        self.offset_proj = nn.Linear(embed_dim, 2)
        
        # Позиционное кодирование
        self.rel_pos_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_heads)
        )
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, C] входные признаки
            positions: [B, N, 2] координаты позиций
            mask: опциональная маска внимания
            
        Returns:
            out: [B, N, C] выходные признаки
            offsets: [B, N, 2] предсказанные смещения
        """
        B, N, C = x.shape
        
        # QKV проекции
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, head_dim]
        
        # Предсказание смещений
        offsets = self.offset_proj(x)  # [B, N, 2]
        offsets = self.deform_scale * torch.tanh(offsets)
        
        # Деформированные позиции
        deformed_positions = positions + offsets  # [B, N, 2]
        
        # Билинейное семплирование ключей и значений
        k_deformed = self.bilinear_sample(k, v, deformed_positions, positions)
        v_deformed = self.bilinear_sample(v, v, deformed_positions, positions)
        
        # Внимание в деформированных координатах
        attn_weights = torch.matmul(q * self.scale, k_deformed.transpose(-2, -1))  # [B, H, N, N]
        
        # Относительное позиционное кодирование в деформированных координатах
        rel_positions = deformed_positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
        rel_pos_encoding = self.rel_pos_embed(rel_positions).permute(0, 3, 1, 2)  # [B, H, N, N]
        attn_weights = attn_weights + rel_pos_encoding
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Применение внимания
        out = torch.matmul(attn_weights, v_deformed)  # [B, H, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        
        return out, offsets
    
    def bilinear_sample(self, features: torch.Tensor, values: torch.Tensor,
                       deformed_positions: torch.Tensor, original_positions: torch.Tensor) -> torch.Tensor:
        """
        Билинейное семплирование признаков в деформированных позициях.
        Реализует субпиксельную точность через интерполяцию первого порядка.
        """
        # Нормализация координат к [-1, 1]
        H = W = int(math.sqrt(features.shape[2]))
        grid_x = deformed_positions[..., 0] / (W - 1) * 2 - 1
        grid_y = deformed_positions[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Reshape для grid_sample
        B, H_attn, N, D = features.shape
        features_reshaped = features.permute(0, 1, 3, 2).reshape(B * H_attn, D, H, W)
        grid_reshaped = grid.reshape(B, N, 2).unsqueeze(1).expand(-1, H_attn, -1, -1).reshape(B * H_attn, N, 2)
        
        sampled = F.grid_sample(
            features_reshaped, 
            grid_reshaped.unsqueeze(2), 
            mode='bilinear', 
            align_corners=True,
            padding_mode='border'
        )
        
        sampled = sampled.squeeze(-1).reshape(B, H_attn, D, N).permute(0, 1, 3, 2)
        return sampled


class DualDeformableAttention(nn.Module):
    """
    Деформируемое внимание с раздельными полями для крон и теней.
    
    Δ_tree(p) = s_tree * tanh(W_offset^(tree) * f(p) + b_offset^(tree))
    Δ_shadow(p) = s_shadow * tanh(W_offset^(shadow) * f(p) + b_offset^(shadow))
                  + W_roof * embed(R(p))
    
    где s_tree = 0.15, s_shadow = 0.45
    """
    
    def __init__(self, embed_dim: int, num_heads: int, num_roof_types: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.s_tree = 0.15
        self.s_shadow = 0.45
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Раздельные поля смещений
        self.offset_tree = nn.Linear(embed_dim, 2)
        self.offset_shadow = nn.Linear(embed_dim, 2)
        
        # Модуляция по типу поверхности крыши
        self.roof_embed = nn.Embedding(num_roof_types, 2)
        self.offset_shadow_mod = nn.Linear(2, 2)
        
        # Позиционное кодирование
        self.rel_pos_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_heads)
        )
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, 
                roof_types: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, C] входные признаки
            positions: [B, N, 2] координаты позиций
            roof_types: [B, N] типы поверхностей крыши (0-3)
            mask: опциональная маска внимания
            
        Returns:
            out: [B, N, C] выходные признаки
            offsets_tree: [B, N, 2] смещения для крон
            offsets_shadow: [B, N, 2] смещения для теней
        """
        B, N, C = x.shape
        
        # QKV проекции
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Предсказание базовых смещений
        offset_base_tree = self.offset_tree(x)  # [B, N, 2]
        offset_base_shadow = self.offset_shadow(x)
        
        # Масштабирование
        offsets_tree = self.s_tree * torch.tanh(offset_base_tree)
        offsets_shadow = self.s_shadow * torch.tanh(offset_base_shadow)
        
        # Модуляция для теней по типу поверхности крыши
        if roof_types is not None:
            roof_embedding = self.roof_embed(roof_types)  # [B, N, 2]
            roof_modulation = self.offset_shadow_mod(roof_embedding)
            offsets_shadow = offsets_shadow + roof_modulation
        
        # Усреднение смещений для основного выхода (можно модифицировать)
        offsets_combined = (offsets_tree + offsets_shadow) / 2
        deformed_positions = positions + offsets_combined
        
        # Билинейное семплирование
        k_deformed = self.bilinear_sample(k, deformed_positions, positions)
        v_deformed = self.bilinear_sample(v, deformed_positions, positions)
        
        # Внимание
        attn_weights = torch.matmul(q * self.scale, k_deformed.transpose(-2, -1))
        
        rel_positions = deformed_positions.unsqueeze(2) - positions.unsqueeze(1)
        rel_pos_encoding = self.rel_pos_embed(rel_positions).permute(0, 3, 1, 2)
        attn_weights = attn_weights + rel_pos_encoding
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v_deformed)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        
        return out, offsets_tree, offsets_shadow
    
    def bilinear_sample(self, features: torch.Tensor, deformed_positions: torch.Tensor,
                       original_positions: torch.Tensor) -> torch.Tensor:
        H = W = int(math.sqrt(features.shape[2]))
        grid_x = deformed_positions[..., 0] / (W - 1) * 2 - 1
        grid_y = deformed_positions[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        B, H_attn, N, D = features.shape
        features_reshaped = features.permute(0, 1, 3, 2).reshape(B * H_attn, D, H, W)
        grid_reshaped = grid.reshape(B, N, 2).unsqueeze(1).expand(-1, H_attn, -1, -1).reshape(B * H_attn, N, 2)
        
        sampled = F.grid_sample(
            features_reshaped,
            grid_reshaped.unsqueeze(2),
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        
        sampled = sampled.squeeze(-1).reshape(B, H_attn, D, N).permute(0, 1, 3, 2)
        return sampled


class CrossAttentionWithSun(nn.Module):
    """
    Перекрёстное внимание между кронами и тенями с учётом направления на солнце.
    
    α_ij^cross = Softmax((q_i^tree · k_j^shadow) / sqrt(d) 
                         + PE(p_j^shadow - p_i^tree) 
                         + PE(θ_sun))
    
    Компонент PE(θ_sun) позволяет учитывать глобальный контекст освещения.
    """
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Позиционное кодирование для относительных позиций
        self.rel_pos_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_heads)
        )
        
        # Кодирование азимута солнца
        self.sun_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),  # sin/cos азимута и угла места
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_heads)
        )
        
    def forward(self, q_tree: torch.Tensor, k_shadow: torch.Tensor, v_shadow: torch.Tensor,
                pos_tree: torch.Tensor, pos_shadow: torch.Tensor,
                sun_azimuth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tree: [B, N_tree, C] запросы от крон
            k_shadow: [B, N_shadow, C] ключи от теней
            v_shadow: [B, N_shadow, C] значения от теней
            pos_tree: [B, N_tree, 2] позиции крон
            pos_shadow: [B, N_shadow, 2] позиции теней
            sun_azimuth: [B, 2] sin/cos азимута и угла места солнца
            
        Returns:
            out: [B, N_tree, C] обновлённые признаки крон
        """
        B, N_tree, C = q_tree.shape
        N_shadow = k_shadow.shape[1]
        
        # Проекции
        q = self.q_proj(q_tree).reshape(B, N_tree, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_shadow).reshape(B, N_shadow, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_shadow).reshape(B, N_shadow, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Внимание
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))  # [B, H, N_tree, N_shadow]
        
        # Относительное позиционное кодирование
        rel_positions = pos_shadow.unsqueeze(1) - pos_tree.unsqueeze(2)  # [B, N_tree, N_shadow, 2]
        rel_pos_encoding = self.rel_pos_embed(rel_positions).permute(0, 3, 1, 2)  # [B, H, N_tree, N_shadow]
        attn_weights = attn_weights + rel_pos_encoding
        
        # Кодирование направления солнца
        sun_encoding = self.sun_embed(sun_azimuth).view(B, self.num_heads, 1, 1)
        attn_weights = attn_weights + sun_encoding
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N_tree, C)
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Блок трансформера с деформируемым вниманием."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 deform_scale: float = 0.15, attention_type: str = 'local',
                 dual_field: bool = False):
        super().__init__()
        self.dual_field = dual_field
        
        if dual_field:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = DualDeformableAttention(embed_dim, num_heads)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = DeformableAttention(embed_dim, num_heads, deform_scale, attention_type)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, 
                roof_types: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.dual_field:
            out, offsets_tree, offsets_shadow = self.attn(
                self.norm1(x), positions, roof_types, mask
            )
            out = out + x
            out = out + self.mlp(self.norm2(out))
            return out, offsets_tree, offsets_shadow
        else:
            out, offsets = self.attn(self.norm1(x), positions, mask)
            out = out + x
            out = out + self.mlp(self.norm2(out))
            return out, offsets, None


class HierarchicalBackbone(nn.Module):
    """
    Иерархический backbone DI-ViT с деформируемым вниманием.
    
    Этапы:
    1. H/4 x W/4 x 64, 2 блока, жёсткое оконное внимание
    2. H/8 x W/8 x 128, 2 блока, локальное деформируемое внимание, Δ_tree
    3. H/16 x W/16 x 256, 4 блока, полное деформируемое внимание, Δ_tree, Δ_shadow
    4. H/32 x W/32 x 512, 2 блока, глобальное деформируемое внимание, Δ_tree, Δ_shadow
    """
    
    def __init__(self, img_size: int = 1024, in_chans: int = 3):
        super().__init__()
        self.img_size = img_size
        
        # Patch embedding для разных стадий
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.patch_embed2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Стадии с разным типом внимания
        # Этап 1: жёсткое оконное внимание (без деформации)
        self.stage1_blocks = nn.ModuleList([
            TransformerBlock(64, num_heads=2, deform_scale=0.0, attention_type='window')
            for _ in range(2)
        ])
        
        # Этап 2: локальное деформируемое внимание, Δ_tree
        self.stage2_blocks = nn.ModuleList([
            TransformerBlock(128, num_heads=4, deform_scale=0.15, attention_type='local')
            for _ in range(2)
        ])
        
        # Этап 3: полное деформируемое внимание, Δ_tree, Δ_shadow
        self.stage3_blocks = nn.ModuleList([
            TransformerBlock(256, num_heads=8, deform_scale=0.15, dual_field=True)
            for _ in range(4)
        ])
        
        # Этап 4: глобальное деформируемое внимание, Δ_tree, Δ_shadow
        self.stage4_blocks = nn.ModuleList([
            TransformerBlock(512, num_heads=16, deform_scale=0.15, dual_field=True)
            for _ in range(2)
        ])
        
        self.norm = nn.LayerNorm(512)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # Этап 1
        x = self.patch_embed1(x)  # [B, 64, H/4, W/4]
        B, C, H, W = x.shape
        positions = self._get_positions(B, H, W, x.device)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        for block in self.stage1_blocks:
            x, _, _ = block(x, positions)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        feat1 = x
        
        # Этап 2
        x = self.patch_embed2(x)  # [B, 128, H/8, W/8]
        B, C, H, W = x.shape
        positions = self._get_positions(B, H, W, x.device)
        x = x.flatten(2).transpose(1, 2)
        
        offsets_tree_list = []
        for block in self.stage2_blocks:
            x, offsets, _ = block(x, positions)
            if offsets is not None:
                offsets_tree_list.append(offsets)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        feat2 = x
        
        # Этап 3
        x = self.patch_embed3(x)  # [B, 256, H/16, W/16]
        B, C, H, W = x.shape
        positions = self._get_positions(B, H, W, x.device)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.stage3_blocks:
            x, offsets_tree, offsets_shadow = block(x, positions)
            if offsets_tree is not None:
                offsets_tree_list.append(offsets_tree)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        feat3 = x
        
        # Этап 4
        x = self.patch_embed4(x)  # [B, 512, H/32, W/32]
        B, C, H, W = x.shape
        positions = self._get_positions(B, H, W, x.device)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.stage4_blocks:
            x, offsets_tree, offsets_shadow = block(x, positions)
            if offsets_tree is not None:
                offsets_tree_list.append(offsets_tree)
        
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        feat4 = x
        
        return {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'offsets_tree': offsets_tree_list if offsets_tree_list else None
        }
    
    def _get_positions(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Генерация сетки позиций."""
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                              torch.arange(W, device=device), indexing='ij')
        positions = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        positions = positions.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        positions = positions.reshape(B, H * W, 2)
        return positions


class RoofGeometryModule(nn.Module):
    """
    Модуль геометрии крыши для классификации типов поверхности.
    
    Классифицирует каждый пиксель как:
    0 - грунт
    1 - плоская крыша
    2 - скат
    3 - конёк
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 512, H/32, W/32] признаки из этапа 4
            
        Returns:
            roof_types: [B, num_classes, H/4, W/4] логиты типов поверхности
        """
        return self.decoder(x)


class DetectionHead(nn.Module):
    """
    Головка детекции для классификации объектов и регрессии bounding boxes.
    
    Предсказывает:
    - Классы объектов (кроны, тени)
    - Координаты рамок (x, y, w, h, φ)
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
        # Классификация
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, num_classes + 1, kernel_size=1)  # +1 для фона
        )
        
        # Регрессия рамок (x, y, w, h, угол)
        self.box_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 5, kernel_size=1)  # x, y, w, h, φ
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] признаки
            
        Returns:
            cls_logits: [B, num_classes+1, H, W] логиты классов
            box_pred: [B, 5, H, W] предсказания рамок
        """
        cls_logits = self.cls_head(x)
        box_pred = self.box_head(x)
        return cls_logits, box_pred


class SunAzimuthHead(nn.Module):
    """Головка оценки солнечного азимута."""
    
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # sin(азимут), cos(азимут)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] признаки
            
        Returns:
            sun_encoding: [B, 2] sin/cos азимута
        """
        return self.head(x)


class ShadowConnectionHead(nn.Module):
    """
    Головка классификации связности пар фрагментов теней.
    
    Если два фрагмента являются частями одной тени, разорванной коньком,
    они получают метку y_ij = 1.
    """
    
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1: [B, C] признаки первого фрагмента
            feat2: [B, C] признаки второго фрагмента
            
        Returns:
            prob: [B, 1] вероятность принадлежности одной тени
        """
        combined = torch.cat([feat1, feat2], dim=1)
        return self.head(combined)


class DIViT(nn.Module):
    """
    Deformation-Invariant Visual Transformer (DI-ViT)
    
    Полная архитектура включает:
    - Иерархический backbone с деформируемым вниманием
    - Четыре параллельные головки (классификация, bbox, азимут солнца, сегментация)
    - Модуль геометрии крыши
    - Деформационно-инвариантный детектор
    """
    
    def __init__(self, img_size: int = 1024, in_chans: int = 3):
        super().__init__()
        self.img_size = img_size
        
        # Backbone
        self.backbone = HierarchicalBackbone(img_size, in_chans)
        
        # Головки
        self.det_head = DetectionHead(in_channels=512, num_classes=2)
        self.sun_head = SunAzimuthHead(in_channels=512)
        self.roof_module = RoofGeometryModule(in_channels=512, num_classes=4)
        self.shadow_connect_head = ShadowConnectionHead(in_channels=512)
        
        # Cross-attention для связывания крон и теней
        self.cross_attn = CrossAttentionWithSun(embed_dim=512, num_heads=16)
        
    def forward(self, x: torch.Tensor, sun_azimuth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] входное изображение
            sun_azimuth: [B, 2] опционально, sin/cos азимута солнца
            
        Returns:
            dict с предсказаниями:
            - cls_logits: логиты классов
            - box_pred: предсказания рамок
            - sun_encoding: оценка азимута солнца
            - roof_types: типы поверхности крыши
            - offsets_tree: поля смещений для крон
            - offsets_shadow: поля смещений для теней
        """
        # Backbone
        features = self.backbone(x)
        feat4 = features['feat4']
        
        # Головки
        cls_logits, box_pred = self.det_head(feat4)
        sun_encoding = self.sun_head(feat4)
        roof_types = self.roof_module(feat4)
        
        return {
            'cls_logits': cls_logits,
            'box_pred': box_pred,
            'sun_encoding': sun_encoding if sun_azimuth is None else sun_azimuth,
            'roof_types': roof_types,
            'offsets_tree': features.get('offsets_tree'),
            'feat4': feat4
        }
    
    def predict_shadow_connection(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Предсказание связности двух фрагментов теней."""
        return self.shadow_connect_head(feat1, feat2)


if __name__ == '__main__':
    # Тест модели
    model = DIViT(img_size=1024, in_chans=3)
    model.eval()
    
    x = torch.randn(2, 3, 1024, 1024)
    sun_azimuth = torch.tensor([[0.5, 0.866], [0.7, 0.714]])  # sin/cos для 2 изображений
    
    with torch.no_grad():
        outputs = model(x, sun_azimuth)
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"    first tensor shape: {value[0].shape}")
