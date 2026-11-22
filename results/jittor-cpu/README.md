# Jittor CPU 复现结果

**作者的两份 8 层 MNIST 权重可以直接载入 Jittor。在相同的四张完整测试图上，两种注意力各自与作者 PyTorch 模型的平均 bits/dim 相差不到 0.000001。** MNIST 与 CIFAR-10 的真实样本也已跑通训练、评估、递推生成和检查点恢复。以下结果对应代码 `56c6495`，运行于本地 WSL Ubuntu 24.04、Intel Core Ultra X7 358H，使用 Jittor 1.3.11.0、PyTorch 2.7.1+cpu、FP32、`OMP_NUM_THREADS=2`、`use_mkl=0`。

## 作者权重在两个框架上的预测

两个框架读取相同的作者权重，每个模型核对 134 个参数或缓冲区条目。在 MNIST 测试集前 4 张图上，以首像素为条件逐位置预测，共评估 4 × 783 个目标。两个框架都按仓库 PyTorch 基线的离散 Logistic 混合似然计算 bits/dim。另从第一张图的首个非零像素取 17 个值，从空状态逐步推理，与作者模型对这一窗口的整段预测比较。[逐项结果](weights.json) · [作者实验与论文 Table 1](https://proceedings.mlr.press/v119/katharopoulos20a.html)

| 注意力 | 作者 PyTorch，bits/dim | 本组 Jittor，bits/dim | 整图 logits 最大绝对误差 | 17 步递推 logits 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 0.493319665 | 0.493320145 | 3.28 × 10⁻⁵ | 5.72 × 10⁻⁶ |
| Causal Linear | 0.532264766 | 0.532264182 | 1.17 × 10⁻⁴ | 7.63 × 10⁻⁶ |

这四张图用于检验权重移植。引用论文 Table 1 或组内模型质量时，应使用完整测试集结果。逐像素负对数似然的最大框架差异分别为 0.000259 和 0.000647 nat，JSON 同时给出了固定相同 logits 后的损失误差。[已有 A30 完整测试集基线](../mnist-a30/quality.json)

## 注意力、训练与数据

- **公式与梯度。** 固定同一组 Q/K/V 和上游梯度，用 PyTorch 显式构造因果权重作为独立参考。在长度 1 和 17、两头、键维 4、值维 3 的样例上，两种注意力的输出最大绝对误差均小于 2.4 × 10⁻⁷，Q/K/V 梯度最大绝对误差为 4.77 × 10⁻⁷。逐步输出与整段输出相差至多 2.09 × 10⁻⁷。改动后半段输入后，前半段输出逐值相同。每一步均检查了 Linear 的 S、Z 形状固定以及 Softmax 的 KV 缓存随步数增长。[数值对齐数据](attention.json)
- **数据与训练。** MNIST 每图 784 个灰度值。CIFAR-10 从文件的三个颜色平面还原为图像，再按行、列、RGB 通道依次展开为 3,072 个值。用手工小图、真实首像素、图像往返还原和错位目标检查顺序。下面每项均取前 2 张训练图更新两次，再取独立测试集前 2 张评估；每张只预测前 192 个目标。配置为 1 层、宽度 32、2 头、10 个混合分量，关闭 dropout。[训练及采样记录](training.json)
- **优化与恢复。** 四次小跑的预测头参数均发生变化，每次都从首像素递推生成 8 个合法像素值。恢复第 1 步检查点后重做第 2 步，四组参数的最大差异均为零。单独用固定梯度跑 8 步 RAdam，与 PyTorch 的参数逐步一致。用 FP64 离散分布直接核对损失，包括 0、255 两个边界，损失及其梯度最大绝对误差分别为 9.85 × 10⁻⁷ 和 4.61 × 10⁻⁷。[训练检查数据](training.json)

| 数据集 | 注意力 | 两次更新前的训练 NLL，nat/目标 | 测试前缀 bits/dim，384 个目标 |
| --- | --- | ---: | ---: |
| MNIST | Softmax | 2.0710 → 2.0706 | 2.7262 |
| MNIST | Causal Linear | 2.0720 → 2.0715 | 2.7277 |
| CIFAR-10 | Softmax | 6.2657 → 6.2656 | 9.2471 |
| CIFAR-10 | Causal Linear | 6.2665 → 6.2664 | 9.2439 |

## 短序列前后向计时

每个长度和方法独立启动进程。对 `[B=2,L,H=2,D=8]` 做一次前向和 Q/K/V 反向。每次使用新输入并同步，计时区间排除输入生成和结果核验。单列第一次调用，再预热 3 次，取后续 7 次中位数。首次调用包含算子准备，耗时受已有 JIT 磁盘缓存影响。[全部计时，毫秒/批](scan.json)

| 序列长度 | Softmax | Causal Linear |
| ---: | ---: | ---: |
| 16 | 0.121 | 0.142 |
| 32 | 0.168 | 0.162 |
| 64 | 0.437 | 0.225 |

短序列上两种方法耗时接近，本机长度 64 时 Linear 用时约为 Softmax 的一半。当前整段 Linear 实现会显式保存各位置的外积前缀，长序列训练的中间张量仍需实测。论文 Figure 1 的长序列、显存和 GPU 性能，以及 Table 1 的训练质量和生成吞吐，将在获卡后按统一配置测量。CPU 记录中的 GPU 显存字段为空。

## 从仓库根目录复跑

以下命令使用 Linux 或 WSL、Python 3.12、g++ 和对应的 Python 开发头文件。建立一个虚拟环境后，安装 CPU 依赖。首次导入 Jittor 会编译后端。[Jittor 安装说明](https://github.com/Jittor/jittor#install)

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r baselines/pytorch/requirements.txt
python -m pip install --no-build-isolation jittor==1.3.11.0
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
export nvcc_path="" use_mkl=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m tests.check_attention --fixtures runs/fixtures --out runs/attention.json
python -m benchmarks.attention_scan --out runs/scan.json
```

`MNIST` 指向四个未压缩 IDX 文件所在目录，`CIFAR` 指向包含 `data_batch_1` 和 `test_batch` 的目录。已有数据可直接设置变量；MNIST 也可用仓库脚本获取，CIFAR-10 选择官网的 Python 版。[CIFAR-10 数据与文件格式](https://www.cs.toronto.edu/~kriz/cifar.html)

```bash
MNIST=data/mnist
CIFAR=data/cifar-10-batches-py
python baselines/pytorch/verify_mnist.py download --data-dir "$MNIST"
python -m tests.check_training --mnist "$MNIST" --cifar10 "$CIFAR" --out runs/training.json
```

复核完整作者权重需要在同一环境构建固定版本的作者 CPU 扩展。以下命令适用于首次准备上游代码的 CPU 环境。[基线来源与校验](../../baselines/pytorch/README.md)

```bash
(
  cd baselines/pytorch
  git -c core.autocrlf=false clone https://github.com/idiap/fast-transformers.git upstream/fast-transformers
  git -C upstream/fast-transformers checkout --detach 2ad36b97e64cb93862937bd21fcc9568d989561f
  python prepare_model.py
  (cd upstream/fast-transformers && MAX_JOBS=1 python setup.py build_ext --inplace)
  python fetch_author_weights.py --out-dir weights
)
python -m tests.check_weights --mnist "$MNIST" --weights baselines/pytorch/weights --out runs/weights.json
```

训练入口可独立使用。下面先在 CPU 上训练 1 轮，再从保存的模型、RAdam 一阶和二阶矩、步数恢复到第 2 轮。本次跨进程复跑与从头连续训练 2 轮的 67 个检查点条目逐项相同。移除 `--tiny` 使用正式宽度和层数，移除 `--targets` 使用完整图像。四组小跑的指标按实际预测目标数归一。

```bash
args=(--dataset mnist --data "$MNIST" --attention causal-linear --tiny --targets 192
      --train-images 2 --test-images 2 --batch 2 --run-dir runs/tiny-mnist)
python -m src.train "${args[@]}" --epochs 1
python -m src.train "${args[@]}" --epochs 2 --resume
```
