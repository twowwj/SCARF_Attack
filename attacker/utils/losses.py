import torch
import torch.nn.functional as F
from typing import Dict, Any

def l_collapse(target_features: torch.Tensor, mu: torch.Tensor, lambda_val: float) -> torch.Tensor:
    """
    计算特征坍塌损失（仅对DC分量）
    Args:
        target_features: 目标特征 (M, D)，M是基元数量，D是特征维度
        mu: 目标向量 (3,)，期望的DC分量均值
        lambda_val: 权重参数，控制均值损失的重要性
    Returns:
        loss: 坍塌损失值，标量tensor
    """
    # 处理空目标区域的情况
    if target_features.shape[0] == 0:
        return torch.tensor(0.0, device=target_features.device, requires_grad=True)
    
    # 确保mu在正确的设备上
    mu = mu.to(target_features.device)
    
    # 只取前3维作为DC分量
    dc_features = target_features[:, :3]  # (M, 3)
    
    # 1. 方差损失：让目标区域内的DC分量变得一致
    # 计算每个DC分量维度的方差，然后求和
    variance_loss = torch.var(dc_features, dim=0).sum()
    
    # 2. 均值损失：让DC分量均值接近目标向量μ
    mean_dc_features = torch.mean(dc_features, dim=0)  # (3,)
    mean_loss = F.mse_loss(mean_dc_features, mu)
    
    # 3. 总损失 = 方差损失 + λ×均值损失
    total_loss = variance_loss + lambda_val * mean_loss
    
    return total_loss

def compute_collapse_metrics(target_features: torch.Tensor, mu: torch.Tensor) -> Dict[str, float]:
    """
    计算特征坍塌的评估指标（仅对DC分量）
    Args:
        target_features: 目标特征 (M, D)
        mu: 目标向量 (3,)
    Returns:
        metrics: 包含各种评估指标的字典
    """
    if target_features.shape[0] == 0:
        return {
            'variance': 0.0,
            'mean_distance': 0.0,
            'collapse_ratio': 0.0,
            'feature_count': 0
        }
    
    # 确保mu在正确的设备上
    mu = mu.to(target_features.device)
    
    # 只取前3维作为DC分量
    dc_features = target_features[:, :3]  # (M, 3)
    
    # 计算方差
    variance = torch.var(dc_features, dim=0).mean().item()
    
    # 计算均值距离
    mean_dc_features = torch.mean(dc_features, dim=0)
    mean_distance = F.mse_loss(mean_dc_features, mu).item()
    
    # 计算坍塌比例（特征标准差相对于原始标准差的比值）
    original_std = torch.std(dc_features, dim=0).mean().item()
    collapse_ratio = 1.0 - (original_std / (original_std + 1e-8))
    
    return {
        'variance': variance,
        'mean_distance': mean_distance,
        'collapse_ratio': collapse_ratio,
        'feature_count': target_features.shape[0]
    }

def compute_detailed_collapse_analysis(target_features: torch.Tensor, mu: torch.Tensor) -> Dict[str, Any]:
    """
    计算详细的特征坍塌分析
    Args:
        target_features: 目标特征 (M, D)
        mu: 目标向量 (D,)
    Returns:
        analysis: 包含详细分析结果的字典
    """
    if target_features.shape[0] == 0:
        return {
            'basic_metrics': compute_collapse_metrics(target_features, mu),
            'feature_statistics': {},
            'convergence_analysis': {}
        }
    
    # 确保mu在正确的设备上
    mu = mu.to(target_features.device)
    
    # 基本指标
    basic_metrics = compute_collapse_metrics(target_features, mu)
    
    # 特征统计
    feature_stats = {
        'mean': torch.mean(target_features, dim=0).detach().cpu().numpy(),
        'std': torch.std(target_features, dim=0).detach().cpu().numpy(),
        'min': torch.min(target_features, dim=0)[0].detach().cpu().numpy(),
        'max': torch.max(target_features, dim=0)[0].detach().cpu().numpy(),
        'target_mu': mu.detach().cpu().numpy()
    }
    
    # 收敛性分析
    mean_features = torch.mean(target_features, dim=0)
    convergence_analysis = {
        'mean_convergence': F.mse_loss(mean_features, mu).item(),
        'variance_convergence': torch.var(target_features, dim=0).mean().item(),
        'max_deviation': torch.max(torch.abs(target_features - mu.unsqueeze(0))).item(),
        'avg_deviation': torch.mean(torch.abs(target_features - mu.unsqueeze(0))).item()
    }
    
    return {
        'basic_metrics': basic_metrics,
        'feature_statistics': feature_stats,
        'convergence_analysis': convergence_analysis
    }

def compute_collapse_loss_components(target_features: torch.Tensor, mu: torch.Tensor, lambda_val: float) -> Dict[str, torch.Tensor]:
    """
    计算坍塌损失的各个组成部分
    Args:
        target_features: 目标特征 (M, D)
        mu: 目标向量 (D,)
        lambda_val: 权重参数
    Returns:
        components: 包含损失组成部分的字典
    """
    if target_features.shape[0] == 0:
        zero_tensor = torch.tensor(0.0, device=target_features.device, requires_grad=target_features.requires_grad)
        return {
            'variance_loss': zero_tensor,
            'mean_loss': zero_tensor,
            'total_loss': zero_tensor
        }
    
    # 确保mu在正确的设备上
    mu = mu.to(target_features.device)
    
    # 方差损失
    variance_loss = torch.var(target_features, dim=0).sum()
    
    # 均值损失
    mean_features = torch.mean(target_features, dim=0)
    mean_loss = F.mse_loss(mean_features, mu)
    
    # 总损失
    total_loss = variance_loss + lambda_val * mean_loss
    
    return {
        'variance_loss': variance_loss,
        'mean_loss': mean_loss,
        'total_loss': total_loss
    }

def salt_and_pepper_loss(image: torch.Tensor, threshold: float = 0.1, temperature: float = 10.0) -> torch.Tensor:
    """
    椒盐噪声loss，鼓励图像中有更多像素接近0或1。
    使用可导的近似函数实现。
    Args:
        image: 输入图像，形状为(C, H, W)或(B, C, H, W)
        threshold: 判定接近0或1的阈值
        temperature: sigmoid函数的温度参数，控制近似精度
    Returns:
        loss: 越小表示椒盐噪声越多
    """
    # 保证输入为4维
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # 使用sigmoid函数实现可导的阈值判断
    # 接近0的像素：sigmoid(-temperature * (image - threshold))
    near_zero = torch.sigmoid(-temperature * (image - threshold))
    # 接近1的像素：sigmoid(temperature * (image - (1 - threshold)))
    near_one = torch.sigmoid(temperature * (image - (1 - threshold)))
    
    # 统计接近0或1的像素比例
    salt_pepper_mask = near_zero + near_one
    salt_pepper_ratio = salt_pepper_mask.mean()
    
    # loss为负的比例，越小越好
    loss = -salt_pepper_ratio
    return loss 