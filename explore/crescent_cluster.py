import numpy as np
import torch


class CrescentClusterer:
    """
    CReSCENT 的结构聚类与内在奖励模块。

    功能：
      - 维护 K 个结构原型（cluster center），维度 = repr_dim
      - 维护每个原型被访问的次数（pseudo-count）
      - 对每一批结构表示 z_seq:
          1) 计算每个 z 的最近中心 index
          2) 基于「旧计数」计算新颖度 r_int
             （当前版本：Gaussian KNN 加权 pseudo-count）
          3) 用 EMA 更新对应的中心
          4) 更新计数
    """

    def __init__(
        self,
        repr_dim: int,
        num_clusters: int = 64,
        ema_momentum: float = 0.99,
        count_smoothing: float = 0.1,
        intrinsic_coef: float = 0.1,
        device: str = "cpu",
        knn_k: int = 4,
        kernel_tau: float = 1.0,
    ):
        """
        参数:
          repr_dim:         结构表示 z 的维度
          num_clusters:     聚类中心数量 K
          ema_momentum:     原型更新的 EMA 系数 m，越接近 1 越平滑
          count_smoothing:  计数平滑常数 c0，用于 1/sqrt(n + c0)
          intrinsic_coef:   内在奖励系数，最终返回的 r_int 会乘上它
          device:           内部张量所在设备
          knn_k:            计算 pseudo-count 时使用的 KNN 个数
          kernel_tau:       Gaussian kernel 的温度系数 tau：
                           w_k ∝ exp(-tau * dist2_k)
        """
        self.repr_dim = repr_dim
        self.num_clusters = num_clusters
        self.ema_momentum = ema_momentum
        self.count_smoothing = count_smoothing
        self.intrinsic_coef = intrinsic_coef
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # KNN + Gaussian kernel 相关超参
        self.knn_k = max(1, int(knn_k))
        self.kernel_tau = float(kernel_tau)

        # 聚类中心与计数
        self.centers = None  # [K, D]，延迟初始化
        self.counts = torch.zeros(num_clusters, dtype=torch.float32, device=self.device)
        self.initialized = False

    # ---------------------------
    # 内部工具函数
    # ---------------------------
    def _to_tensor(self, z):
        """把输入 z（np.ndarray 或 torch.Tensor）转成 [N, D] 的 float32 tensor。"""
        if isinstance(z, np.ndarray):
            z_t = torch.from_numpy(z)
        else:
            z_t = z
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(self.device, dtype=torch.float32)
        return z_t

    def _init_centers(self, z_batch: torch.Tensor):
        """
        使用第一批 z 初始化聚类中心。
        简单策略：从 batch 中随机采样 K 个样本作为初始 center。
        """
        N = z_batch.size(0)
        K = self.num_clusters

        if N >= K:
            # 随机打乱后取前 K 个
            perm = torch.randperm(N, device=self.device)
            select = perm[:K]
            centers = z_batch[select].clone()
        else:
            # 样本太少，重复采样补足
            import math  # 局部导入以避免不必要的全局依赖
            repeat_factor = int(math.ceil(K / max(N, 1)))
            tiled = z_batch.repeat(repeat_factor, 1)  # [repeat_factor*N, D]
            centers = tiled[:K].clone()

        self.centers = centers  # [K, D]
        self.initialized = True

    def _assign_clusters(self, z: torch.Tensor):
        """
        对每个 z 分配最近的中心。
        输入:
          z: [T, D]
        输出:
          idx: [T]，每个样本的最近中心 index
          dist2: [T, K]，平方距离
        """
        # z: [T, D], centers: [K, D]
        T, D = z.size()
        K, D2 = self.centers.size()
        assert D == D2

        # 使用欧氏距离的平方：||z||^2 + ||c||^2 - 2 z·c
        z2 = (z ** 2).sum(dim=1, keepdim=True)              # [T, 1]
        c2 = (self.centers ** 2).sum(dim=1).unsqueeze(0)    # [1, K]
        dist2 = z2 + c2 - 2.0 * (z @ self.centers.t())      # [T, K]
        idx = dist2.argmin(dim=1)  # [T]
        return idx, dist2

    def _update_centers_and_counts(self, z: torch.Tensor, idx: torch.Tensor):
        """
        使用本批 z 对聚类中心和计数进行更新。
        更新规则：
          centers[k] <- m * centers[k] + (1-m) * mean(z in cluster k)
          counts[k]  <- counts[k] + n_k
        """
        K = self.num_clusters
        m = self.ema_momentum

        for k in range(K):
            mask_k = (idx == k)
            n_k = mask_k.sum().item()
            if n_k <= 0:
                continue
            z_k = z[mask_k]           # [n_k, D]
            z_mean = z_k.mean(dim=0)  # [D]

            # EMA 更新
            self.centers[k].data.mul_(m).add_(z_mean.data, alpha=1.0 - m)
            # 累积计数
            self.counts[k] += float(n_k)

    # ---------------------------
    # 对外主接口
    # ---------------------------
    def update_and_compute_intrinsic(self, z_seq):
        """
        输入一条轨迹对应的结构表示序列 z_seq，更新聚类状态并返回内在奖励序列。

        参数:
          z_seq: [T, D] 的 np.ndarray 或 torch.Tensor

        返回:
          r_int_seq: np.ndarray, shape [T]，每个时间步的内在奖励
        """
        # [T, D]
        z = self._to_tensor(z_seq)
        T, D = z.size()
        if T == 0:
            return np.zeros((0,), dtype=np.float32)

        # 第一次调用时，初始化中心
        if not self.initialized:
            self._init_centers(z)

        old_counts = self.counts.clone()  # [K]

        # cluster assignment
        idx, dist2 = self._assign_clusters(z)  # idx: [T], dist2: [T, K]

        # ---------- Gaussian KNN 加权 pseudo-count ----------
        # dist2: [T, K]，对每个时间步 t 取 K 个最近的簇
        K_eff = min(self.num_clusters, self.knn_k)
        # 取每一行最小的 K_eff 个距离及其 index
        dist2_knn, idx_knn = torch.topk(
            dist2, k=K_eff, dim=1, largest=False
        )  # [T, K_eff], [T, K_eff]
        counts_knn = old_counts[idx_knn]  # [T, K_eff]

        # Gaussian kernel 权重：w ∝ exp(-tau * dist2)
        weights = torch.exp(-self.kernel_tau * dist2_knn)  # [T, K_eff]
        # 避免除 0
        eps = 1e-8
        weights_sum = weights.sum(dim=1, keepdim=True) + eps
        weights = weights / weights_sum

        # 加权 pseudo-count：soft_count = Σ w_k * count_k
        soft_count = (weights * counts_knn).sum(dim=1)  # [T]

        # 新颖度：1 / sqrt(soft_count + c0)
        novelty = 1.0 / torch.sqrt(soft_count + self.count_smoothing)  # [T]

        # EMA 更新中心 & 累积计数（仍然使用硬分配 idx）
        self._update_centers_and_counts(z, idx)

        # 乘上内在奖励系数
        r_int = self.intrinsic_coef * novelty  # [T]

        return r_int.detach().cpu().numpy().astype(np.float32)
