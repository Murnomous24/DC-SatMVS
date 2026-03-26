import torch
import torch.nn.functional as F


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

def get_soft_histogram(depth, mask, bins):
    """
    Args:
        depth: [H, W] 或 [B, H, W] 深度图
        mask: [H, W] 或 [B, H, W] 有效区域掩码
        bins: [M_bins] 直方图 bin 的中心点
    Returns:
        hist: [M_bins] 归一化的概率分布
    """
    # 展平以便计算
    d = depth[mask]  # [N]
    if d.numel() == 0:
        return torch.ones_like(bins) / bins.numel()  # 防止空 mask

    # 计算间距
    delta = bins[1] - bins[0]
    
    # 计算距离 [N, M_bins]
    diff = torch.abs(d.unsqueeze(1) - bins.unsqueeze(0))
    
    # 软分配：三角形核函数 (Soft assignment)
    weights = torch.clamp(1.0 - diff / delta, min=0.0)
    
    # 累加每个 bin 的权重
    hist = weights.sum(dim=0)
    
    # 归一化为概率分布 (Sum to 1)
    return hist / (hist.sum() + 1e-8)

def depth_distribution_similarity_loss(depth, depth_gt, mask):
    """
    Args:
        depth: 预测的深度图
        depth_gt: GT 深度图
        mask: 有效区域掩码
    """
    device = depth.device
    M_bins = 48

    # 动态计算 bins 范围（原始方式）
    kl_min = torch.min(depth_gt.min(), depth.mean() - 3. * depth.std()).item()
    kl_max = torch.max(depth_gt.max(), depth.mean() + 3. * depth.std()).item()
    bins = torch.linspace(kl_min, kl_max, steps=M_bins, device=device)

    # 1. 获取分布 (对应公式 2, 3)
    pred_dist = get_soft_histogram(depth, mask, bins)
    gt_dist = get_soft_histogram(depth_gt, mask, bins)

    # 2. 计算 KL 散度 (对应公式 4)
    # F.kl_div(input, target) = target * (log(target) - input)
    # 论文中 KL(d', d_gt') 即 KL(预测 || 真实)
    # 按照 PyTorch 定义: input = log(预测), target = 真实
    # 注意：论文公式(4)是对所有 bin 求和，所以这里用 reduction='sum'
    kl_div = F.kl_div(torch.log(pred_dist + 1e-8), gt_dist, reduction='sum')

    # 3. 分段函数处理 (对应公式 5)
    # Loss = log(kl) if kl > 1 else 0
    # 使用 ReLU 变体实现分段逻辑，保证梯度连续性
    if kl_div > 1.0:
        loss_dcl = torch.log(kl_div)
    else:
        loss_dcl = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_dcl

def STsatmvsloss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    dds_loss_stages = []
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_fre = stage_inputs["depth_filtered"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        dds_loss = depth_distribution_similarity_loss(depth_fre, depth_gt, mask)
        dds_loss_stages.append(dds_loss)

        # total loss
        lam1, lam2 = 0.8, 0.2
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (lam1 * depth_loss + lam2 * dds_loss)
        else:
            total_loss += 1.0 * (lam1 * depth_loss + lam2 * dds_loss)

    return total_loss, depth_loss