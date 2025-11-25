import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureEncoder(nn.Module):
    """
    将宏观结构特征向量 x_t 映射到潜在表示 z_t 的简单编码器：
        z_t = f_theta(x_t)

    设计原则：
      - 输入：任意维度的一维特征向量 macro_feat，形状 [B, input_dim]
      - 输出：结构表示 z，形状 [B, repr_dim]
      - 结构：2 层 MLP + LayerNorm，尽量轻量、稳定
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 1,
    ):
        """
        参数:
          input_dim:  输入特征维度 (macro_feat_dim)
          repr_dim:   输出结构表示维度
          hidden_dim: 隐层宽度
          num_hidden_layers: 隐层层数 (目前只支持 1 或 2，够用了)
        """
        super().__init__()

        if num_hidden_layers not in (1, 2):
            raise ValueError("num_hidden_layers 目前只支持 1 或 2")

        layers = []

        # 第一层: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 可选第二个隐藏层: hidden_dim -> hidden_dim
        if num_hidden_layers == 2:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 输出层: hidden_dim -> repr_dim
        layers.append(nn.Linear(hidden_dim, repr_dim))

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(repr_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
          x: [B, input_dim] 的宏观结构特征向量

        输出:
          z: [B, repr_dim] 的结构表示
        """
        # 确保是 float32
        if x.dtype != torch.float32:
            x = x.float()

        z = self.net(x)
        z = self.norm(z)
        return z


class StructureProjHead(nn.Module):
    """
    对比学习投影头：
        h = g_phi(z)

    一般做法是：
      - encoder 输出的 z 主要给后续模块用（比如聚类 / intrinsic reward）
      - 投影头输出的 h 只在 InfoNCE / 对比损失里使用

    这里用一个简单的 2 层 MLP + ReLU + LayerNorm。
    """

    def __init__(
        self,
        repr_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = None,
    ):
        """
        参数:
          repr_dim: encoder 输出 z 的维度
          proj_dim: 投影空间维度 (用于对比学习)
          hidden_dim: 中间层维度，默认等于 repr_dim
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = repr_dim

        self.mlp = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.norm = nn.LayerNorm(proj_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入:
          z: [B, repr_dim]

        输出:
          h: [B, proj_dim]，对比空间中的表示，一般会在外面再做归一化
        """
        if z.dtype != torch.float32:
            z = z.float()

        h = self.mlp(z)
        h = self.norm(h)
        h = F.normalize(h, dim=-1)
        return h
