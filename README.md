# C3D_FBDConv_EMHSA
1. 代码概述
    (1)本仓库在 mmaction2 框架下实现了一个强化版的 C3D 骨干网络——C3D + FBDConv + EMHSA。
    (2)C3D：原始 3D 卷积骨干，用于提取视频的时空特征。
    (3)FBDConv（Filter-Based Dynamic Convolution）：替换了原 C3D 中两层标准 3×3×3 卷积，通过动态生成卷积核权重来增强通道与空间感受野。
    (4)EMHSA（Efficient Multi-Head Self-Attention）：在主干网络最后插入高效多头自注意力模块，结合深度可分离卷积对特征进行自适应融合，进一步提升表征能力。

2. 核心模块
  (1)FBDConv：
    利用 Attention3D 生成通道、滤波、空间和核选择注意力权重；
    按批次动态聚合多组基础卷积核，实现输入自适应的 3×3×3 卷积。
  (2)EMHSA：
    将最后一层输出展平为序列，使用 nn.MultiheadAttention 捕捉长程依赖；
    序列重塑为 3D 体积后，通过 3×3×3 深度卷积＋1×1×1 逐点卷积完成特征融合；
    还原为时空特征图并送入全连接层完成后续分类或回归。

3. 快速上手
  (1)将本代码文件重命名为 c3d.py；
  (2)放置到 mmaction2/mmaction/models/backbones/c3d.py 路径，覆盖原有实现；
  (3)按照 mmaction2 官方文档安装依赖并执行训练/测试脚本，即可在标准 mmaction2 框架下无缝运行。
