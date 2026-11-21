# 用 Jittor 复现因果线性注意力

当生成一张图需要逐个预测 784 个像素时，模型如何利用已经生成的前缀？本项目复现 ICML 2020 论文 *Transformers are RNNs*。我们将实现两种自回归模型，比较 Softmax 注意力与维护固定大小状态的因果线性注意力在生成质量、耗时和显存上的差异。[论文](https://proceedings.mlr.press/v119/katharopoulos20a.html) · [作者实现](https://github.com/idiap/fast-transformers)

| 实验 | 目标与进度 |
| --- | --- |
| PyTorch 实验基线 | 已完成。复评作者权重，训练两组 MNIST 模型，测量逐像素生成吞吐和注意力的长度曲线。 |
| Jittor 主复现 | 计划实现因果线性注意力、递推推理及 Softmax 对照。Figure 1 比较长度 512–65,536 的前后向耗时和显存。Table 1 比较 MNIST 的 bits/dim 和图/秒，两个模型使用相同训练预算。 |
| 两项额外工作 | 在 CIFAR-10 上以相同训练时长比较生成质量；在同一张卡上对比 Jittor/PyTorch 的性能，定位瓶颈并验证优化效果。 |
| 后续拓展 | 在上述实验完成后，研究有限显存下的前缀状态检查点选择，用实测的存取和恢复耗时修正成本估计。 |

## 从基线开始

[基线运行说明](baselines/pytorch/README.md)提供从安装、数据准备到训练、评估及测速的命令。只想看结果时，可直接打开[逐轮质量](results/mnist-a30/quality.json)、[逐次生成计时](results/mnist-a30/generation.json)和[长度扫描](results/mnist-a30/attention-scaling.json)。

| MNIST 指标 | Softmax | Causal Linear |
| --- | ---: | ---: |
| 原论文 Table 1，250 轮，bits/dim ↓ | 0.621 | 0.644 |
| 作者公开权重，本组复评，bits/dim ↓ | 0.6142 | 0.6376 |
| 本组训练 5 轮，双种子均值，bits/dim ↓ | 0.9562 | 0.9807 |
| 本组递推生成，batch 1，图/秒 ↑ | 0.644 | 0.565 |
| 本组递推生成，batch 500，图/秒 ↑ | 39.343 | 199.764 |

本组使用单张 A30 24 GB、PyTorch 2.7.1、FP32。质量评估覆盖一万张测试图，以首像素为条件，对其余 783 个预测像素归一。固定第 5 轮取种子 1229、3407 的均值。表中速度来自种子 1229 的第 5 轮权重，比较 Softmax KV cache 与 Linear RNN：每批从空状态开始，预热 3 次，每轮计时 10 次，两轮合并取中位数。batch 1 时 Softmax 略快，batch 500 时 Linear 的吞吐约为其 5.08 倍。

长度扫描中，序列长 8,192 时，Full 和 Linear 的前后向耗时分别为 99.365 和 11.506 毫秒/序列。Full 从 16,384 起超过本次显存预算，Linear 完成了 65,536。逐轮数据保留了全部结果，包括 Full 在种子 1229 第 4 轮的波动。

## 目录

- `baselines/pytorch/` 存放已运行的对照程序，作者模型和算子从固定来源获取。本组代码负责离散似然损失、实验入口与计时。
- `results/mnist-a30/` 保存上述 PyTorch 实验的可复核数据。Jittor 实现落地时放入 `src/`，数值对齐用例放入 `tests/`，跨框架实验入口放入 `benchmarks/`。

本组代码使用 [MIT 许可证](LICENSE)。作者库沿用其[上游许可证](https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/LICENSE)，作者 notebook、权重与数据分别由原发布方提供。
