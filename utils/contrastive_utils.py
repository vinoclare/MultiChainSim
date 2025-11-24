import torch


def compute_contrastive_loss(
    h: torch.Tensor,
    episode_ids,
    step_ids,
    time_window: int = 1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    基于投影向量 h 计算一个简单的 InfoNCE 对比损失。

    正样本定义:
      - 同一条 episode
      - 时间步差 |t_i - t_j| <= time_window
      - i != j

    负样本定义:
      - batch 内除了正样本和自身以外的所有样本

    参数:
      h: [B, proj_dim]，已 L2-normalize 的对比表示
      episode_ids: [B]，每个样本所属的 episode id（可以是 list / numpy / torch）
      step_ids: [B]，每个样本在该 episode 内的 step 索引
      time_window: 正样本时间窗口大小
      temperature: InfoNCE 温度参数 τ

    返回:
      loss: 标量 tensor，mean over 有正样本的 anchor
    """
    device = h.device
    B = h.size(0)
    if B <= 1:
        # batch 太小，没法做对比，直接返回 0
        return torch.tensor(0.0, device=device, dtype=h.dtype)

    # 转成 tensor
    ep = torch.as_tensor(episode_ids, device=device)
    ts = torch.as_tensor(step_ids, device=device)

    # 相似度矩阵: [B, B]
    # 假设 h 已经经过 F.normalize
    sim = h @ h.t()  # cosine similarity，因为 h 已经归一化

    # 温度缩放
    sim = sim / temperature

    # 去掉对角线 (self-similarity)
    # 不直接改 sim，使用 mask 控制
    eye = torch.eye(B, dtype=torch.bool, device=device)

    # 正样本 mask: 同 episode 且 时间相近 且 非自身
    same_ep = ep.unsqueeze(0) == ep.unsqueeze(1)         # [B, B]
    dt = (ts.unsqueeze(0) - ts.unsqueeze(1)).abs()       # [B, B]
    close_time = dt <= time_window
    pos_mask = same_ep & close_time & (~eye)             # [B, B]

    # 负样本 mask: 不是正样本且不是自己
    neg_mask = (~pos_mask) & (~eye)

    # 如果某一行没有任何正样本，则该行不纳入 loss
    has_pos = pos_mask.any(dim=1)                        # [B]
    if not has_pos.any():
        return torch.tensor(0.0, device=device, dtype=h.dtype)

    # 为数值稳定做行内减 max
    row_max, _ = sim.max(dim=1, keepdim=True)            # [B, 1]
    logits = sim - row_max                               # [B, B]

    # 只对非自身位置生效
    valid_mask = (~eye)

    # exp(logits) 但只对 valid 位置
    exp_logits = torch.exp(logits) * valid_mask          # [B, B]

    # 正样本分子: sum_j exp(logit_ij) over j in P(i)
    pos_exp = exp_logits * pos_mask                      # [B, B]
    pos_sum = pos_exp.sum(dim=1)                         # [B]

    # 分母: sum_k exp(logit_ik) over k != i
    denom = exp_logits.sum(dim=1)                        # [B]

    eps = 1e-8
    # 只对有正样本的 i 计算 loss
    pos_sum = pos_sum[has_pos]
    denom = denom[has_pos]

    loss_vec = -torch.log((pos_sum + eps) / (denom + eps))
    loss = loss_vec.mean()

    return loss
