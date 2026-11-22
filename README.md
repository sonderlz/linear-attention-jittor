# 用 Jittor 复现因果线性注意力

当生成一张图需要逐个预测 784 个像素时，模型如何利用已经生成的前缀？本项目复现 ICML 2020 论文 *Transformers are RNNs*，用 Jittor 比较 Softmax 注意力与维护固定大小状态的因果线性注意力在生成质量、耗时和显存上的差异。[论文](https://proceedings.mlr.press/v119/katharopoulos20a.html) · [作者实现](https://github.com/idiap/fast-transformers)

| 实验 | 目标与进度 |
| --- | --- |
| PyTorch 实验基线 | 已完成。复评作者权重，训练两组 MNIST 模型，测量逐像素生成吞吐和注意力的长度曲线。 |
| Jittor 主复现 | 两种注意力、递推生成、MNIST 小模型训练和作者权重对齐已在 CPU 上通过。获卡后复现 Figure 1 的长序列耗时与显存，以及 Table 1 的质量和生成吞吐。 |
| 两项额外工作 | CIFAR-10 已通过真实图像的编码与训练小跑。获卡后按相同训练时长比较生成质量，并在同一张卡上对照 Jittor/PyTorch 的性能、定位瓶颈和验证优化效果。 |
| 后续拓展 | 在上述实验完成后，研究有限显存下的前缀状态检查点选择，用实测的存取和恢复耗时修正成本估计。 |

## Jittor CPU 结果

用作者发布的两份 8 层权重逐位置核对 MNIST 测试集前 4 张图，共 3,132 个预测目标。Softmax 和 Causal Linear 的 Jittor 平均 bits/dim 分别与作者 PyTorch 实现相差 4.80 × 10⁻⁷ 和 5.85 × 10⁻⁷。独立算子检查覆盖前向、Q/K/V 梯度、因果掩码与递推状态。两种注意力在 MNIST、CIFAR-10 上均已跑通短前缀训练、评估和恢复。[CPU 实验报告、原始结果与复跑命令](results/jittor-cpu/README.md)

## PyTorch 基线

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
- `src/` 包含 Jittor 模型、图像数据接口、RAdam 优化器及训练入口 `python -m src.train`。
- `tests/` 包含算子、真实数据训练与作者权重的数值验证。`benchmarks/` 包含分进程的注意力前后向长度扫描。
- `results/` 保存 [PyTorch A30 基线](results/mnist-a30/)与 [Jittor CPU 实测](results/jittor-cpu/)。

本组代码使用 [MIT 许可证](LICENSE)。作者库沿用其[上游许可证](https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/LICENSE)，作者 notebook、权重与数据分别由原发布方提供。
