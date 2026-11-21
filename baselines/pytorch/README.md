# PyTorch 基线的运行方法

以下 Bash 命令从仓库根目录开始。A30 测量使用 Linux、Python 3.12、PyTorch 2.7.1+cu128 和两个 CPU 线程。构建作者的 C++/CUDA 扩展需要匹配的 CUDA Toolkit（含 nvcc）、C++ 编译器和 Python 开发头文件。

## 安装并核对模型

```bash
cd baselines/pytorch
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
git -c core.autocrlf=false clone https://github.com/idiap/fast-transformers.git upstream/fast-transformers
git -C upstream/fast-transformers checkout --detach 2ad36b97e64cb93862937bd21fcc9568d989561f
python prepare_model.py
python source_check.py
(cd upstream/fast-transformers && MAX_JOBS=1 python setup.py build_ext --inplace)
python check_install.py --device cuda
```

`prepare_model.py` 从[作者 Colab](https://colab.research.google.com/drive/1BV4OaWRAHYGeimO0cqHD86GfnfgUaKD4)读取原实验用的模型与采样函数。也可下载作者 notebook 后运行 `python prepare_model.py --notebook /path/to/author_mnist.ipynb`。SHA-256 清单固定了上游源码、notebook 和抽取结果。最后一条命令检查编译算子的前向与梯度，并检验同一模型在并行计算和逐步推理下的输出是否一致。

## 用作者权重评估

```bash
python verify_mnist.py download --data-dir data/mnist
python fetch_author_weights.py --out-dir weights
python run_mnist.py evaluate --device cuda --threads 2 --attention full \
  --weights weights/full_author.pt --data-dir data/mnist --batch-size 10 --out runs/author-full.json
python run_mnist.py evaluate --device cuda --threads 2 --attention causal-linear \
  --weights weights/linear_author.pt --data-dir data/mnist --batch-size 10 --out runs/author-linear.json
```

已有四个原始 IDX 文件时可用 `verify_mnist.py verify --data-dir data/mnist` 核对，无需再下载。评估文件的 `metrics.bpd_predicted_783` 对应结果表。预训练权重取自作者，与本组从头训练的权重可通过 SHA-256 区分。

## 复跑 5 轮双种子训练

从一个新的输出目录开始。以下命令按 epoch 续训并在完整测试集上评估，结果表使用固定的第 5 轮权重。训练固定 batch=10、RAdam、学习率 1e-4，Full 和 Linear 的训练预算相同。

```bash
set -e
for seed in 1229 3407; do
  for model in full causal-linear; do
    run="runs/s${seed}-${model}"
    for epoch in 1 2 3 4 5; do
      resume=()
      if (( epoch > 1 )); then resume=(--resume "$run/last.pt"); fi
      python run_mnist.py train --device cuda --threads 2 --attention "$model" \
        --seed "$seed" --batch-size 10 --lr 0.0001 --epochs "$epoch" \
        --data-dir data/mnist --run-dir "$run" "${resume[@]}"
      python run_mnist.py evaluate --device cuda --threads 2 --batch-size 10 \
        --weights "$run/last.pt" --data-dir data/mnist --out "$run/test-e${epoch}.json"
    done
  done
done
```

## 公平测速

两种递推模型在同一块卡上按相同 batch 生成完整图像。以下命令复跑首页两行吞吐结果，也覆盖 batch 10、128 和小 batch 下的无缓存对照。每种配置预热 3 次后计时 10 次，第二轮反转方法执行顺序。每条记录用 batch 除以同步墙钟秒数，合并两轮的 20 个值取中位数。记录包含采样耗时，每批重新初始化模型状态。

```bash
for round in 1 2; do
  for batch in 1 10 128 500; do
    methods=(linear full_kv)
    if (( batch <= 10 )); then methods+=(full_uncached); fi
    for ((i=0; i<${#methods[@]}; i++)); do
      j=$i
      if (( round == 2 )); then j=$((${#methods[@]} - 1 - i)); fi
      case "${methods[j]}" in
        linear)        model=causal-linear; mode=recurrent ;;
        full_kv)       model=full; mode=recurrent ;;
        full_uncached) model=full; mode=parallel ;;
      esac
      python run_mnist.py benchmark --threads 2 --device cuda --seed 1229 \
        --weights "runs/s1229-${model}/last.pt" --modes "$mode" --batch-sizes "$batch" \
        --warmup 3 --repeats 10 --allocator-cache warm \
        --out "runs/gen-${methods[j]}-b${batch}-r${round}.json"
    done
  done
done
```

长度扫描只计因果注意力的前向和 Q/K/V 梯度，按序列归一。原记录取两轮，每轮预热 3 次、计时 5 次。两种方法在每个长度取相同 batch，显存预算为单卡的 80%，OOM 配置也保存在结果中。

```bash
for round in 1 2; do
  methods=(causal-linear full)
  if (( round == 2 )); then methods=(full causal-linear); fi
  for n in 512 1024 2048 4096 8192 16384 32768 65536; do
    batch=$((8192 / n))
    if (( batch < 1 )); then batch=1; fi
    for model in "${methods[@]}"; do
      python benchmark_attention.py --attention "$model" --length "$n" --batch "$batch" \
        --warmup 3 --repeats 5 --memory-fraction 0.8 --out "runs/attn-${model}-${n}-r${round}.json"
    done
  done
done
```
