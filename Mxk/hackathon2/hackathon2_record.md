最后更新：2026-07-27（499 机已关机，数据盘保留）

# Hackathon 2 — 训练记录与结果

## 环境
- **平台**：AutoDL，北京 B 区
- **GPU**：RTX 5090 (32GB) × 2（499机，已关机）
- **框架**：nanoinfra（editable install），路径 `/root/autodl-tmp/nanoinfra/`
- **数据**：FineWeb sample-10BT shard 003/004/005/006
- **旧 SSH**：`wsl ssh -p 35000 root@connect.bjb1.seetacloud.com`
- **499 机 SSH**：`wsl ssh -p 17790 root@connect.bjb2.seetacloud.com`
- **Python**：`/root/miniconda3/bin/python3`

最后更新：2026-07-27（499 机已关机，数据盘保留）

---

---

## Round 1：vocab=32768（默认 tokenizer），d10，lr=6e-4

```
python -m modalities.text.train_text \
  model.depth=10 \
  max_steps=3600 \
  sequence_len=2048 \
  device_batch_size=2 \
  total_batch_size=131072 \
  use_compile=true \
  logging.log_every=25 \
  evaluation.text.interval_steps=300 \
  evaluation.text.eval_tokens=1048576 \
  optimizer.lr_max=6e-4 \
  checkpoint.enabled=true \
  checkpoint.save_dir=/root/autodl-tmp/nanoinfra/models/hackathon2/run1 \
  checkpoint.save_every=2000 \
  checkpoint.keep_last_n=1 \
  seed=42
```

| 参数 | 值 |
|---|---|
| depth | 10 |
| dim | 640 |
| n_head | 5 |
| vocab_size | 32768 |
| 参数量 | 0.09B |
| seq_len | 2048 |
| device_batch_size | 2 |
| total_batch_size | 131072 |
| Grad accum steps | 32 |
| scheduler | linear + 20% warmdown |
| compile | true |
| liger-kernel | true |
| throughput | ~110-130K tok/s |
| MFU | ~31-36% |
| 总运行时间 | ~39 min (stopped at step 2100) |

### 结果（val BPB 变化）

| 时间 | step | train loss | val CE | val BPB |
|---|---|---|---|---|
| 0 min | 0 | 10.40 | 10.09 | 3.27 |
| 6 min | 300 | 5.00 | 4.98 | 1.613 |
| 12 min | 600 | 4.54 | 4.51 | 1.462 |
| 17 min | 900 | 4.29 | 4.30 | 1.395 |
| 23 min | 1200 | 4.19 | 4.19 | 1.359 |
| 28 min | 1500 | 4.12 | 4.12 | 1.334 |
| 34 min | 1800 | 4.05 | 4.07 | 1.318 |
| 39 min | 2100 | 4.04 | 4.03 | 1.308 |

Checkpoint：`/root/autodl-tmp/nanoinfra/models/hackathon2/run1/step_002000`

---

## Round 2：vocab=65536（重训 tokenizer），d10，lr=6e-4

Tokenizer 重训：
```
python -m modalities.text.train_tokenizer \
  --vocab-size 65536 --force --train-chars 80000000 \
  --out /root/autodl-tmp/nanoinfra/outputs/tokenizer
```

训练命令同 Round 1，checkpoint 路径改为 `run2_vocab65536`，max_steps=2100。

| 参数 | 值 |
|---|---|
| vocab_size | 65536 |
| 参数量 | 0.13B（+44%） |
| throughput | ~68-75K tok/s（-40%） |
| MFU | ~23-26% |
| 总运行时间 | ~39 min，在 step 1225 处因数据耗尽停止 |

### 结果

| 时间 | step | train loss | val CE | val BPB |
|---|---|---|---|---|
| 0 min | 0 | 11.09 | 10.74 | 3.33 |
| 10 min | 300 | 5.15 | 5.12 | 1.586 |
| 20 min | 600 | 4.75 | 4.70 | 1.458 |
| 29 min | 900 | 4.48 | 4.47 | 1.386 |
| 39 min | 1200 | 4.39 | 4.36 | 1.351 |

Checkpoint：仅 step_000000

---

## 结论

- Round 1 (vocab=32768) **完胜**：30 min 处 BPB **1.33** vs Round 2 的 **~1.39**
- 大 vocab 的参数爆炸（0.09B → 0.13B）导致吞吐下降 40%，模型学到更少数据
- 虽然大 vocab 的 token/byte 比理论上对 BPB 有利，但在短训练窗口内被吞吐下降完全抵消
- **Hackathon 2 最优方向**：小 vocab + 深模型 + 最大吞吐

---

## Round 3a：2×5090, d14, bs=2（FAIL）

```
torchrun --nproc_per_node=2 -m modalities.text.train_text \
  model.depth=14 device_batch_size=2 total_batch_size=131072 \
  max_steps=2100 lr_max=6e-4 ...
```

| 参数 | 值 |
|---|---|
| depth | 14 |
| dim | 896 |
| 参数量 | 0.19B |
| throughput | ~86K tok/s |
| MFU | ~26% |
| 原因 | batch 太小，FSDP 通信开销大 |

- **吞吐 ~86K tok/s，比 Round 1 单卡 (~120K) 还慢！**
- 问题：device_batch_size=2 导致 grad accum=16，kernel 太小，通信占主导

## Round 3b：2×5090, d12, bs=8（★ 最佳）

```
torchrun --nproc_per_node=2 -m modalities.text.train_text \
  model.depth=12 device_batch_size=8 total_batch_size=131072 \
  max_steps=3600 lr_max=6e-4 ...
```

| 参数 | 值 |
|---|---|
| depth | 12 |
| dim | 768 |
| n_head | 6 |
| vocab_size | 32768 |
| 参数量 | 0.14B |
| seq_len | 2048 |
| device_batch_size | 8 |
| total_batch_size | 131072 |
| Grad accum | 4 |
| throughput | ~285K tok/s |
| MFU | ~60% |
| 总 tokens 处理 | ~510M (30 min) |
| 总训练时间 | 29.2 min (step 3599) |

### 结果（val BPB 变化）

| 时间 | step | train loss | val CE | val BPB |
|---|---|---|---|---|
| 0 min | 0 | 10.40 | 10.06 | 3.2755 |
| 5 min | 300 | 4.99 | 4.96 | 1.6155 |
| 10 min | 600 | 4.51 | 4.51 | 1.4676 |
| 15 min | 900 | 4.27 | 4.27 | 1.3910 |
| 20 min | 1200 | 4.13 | 4.15 | 1.3527 |
| 25 min | 1500 | 4.07 | 4.07 | 1.3266 |
| 30 min | 1800 | 4.01 | 4.03 | **1.3116** |
| 35 min? | 2100 | ~3.96 | - | - |
| final (29.2min) | 3599 | 3.66 | 3.7068 | **1.1993** |

最终 eval：29.2 分钟 checkpoint → **BPB 1.1993**（loaded single-GPU eval）

## 对比总结

| Round | GPUs | Config | Tok/s | 30min BPB |
|---|---|---|---|---|
| 1 | 1×5090 | d10, vocab=32K, bs=2 | ~120K | 1.334 |
| 3b | 2×5090 | d12, vocab=32K, bs=8 | **~285K** | **1.1993** |

**2×5090 比 1×5090 BPB 改善 0.135！** 🎉

## Round 4：2×5090, d12, bs=16（吞吐进一步优化）

```
torchrun --nproc_per_node=2 -m modalities.text.train_text \
  model.depth=12 device_batch_size=16 total_batch_size=131072 \
  max_steps=5100 lr_max=6e-4 ...
```

| 参数 | 值 |
|---|---|
| depth | 12 |
| dim | 768 |
| device_batch_size | **16** |
| Grad accum | **2**（vs bs=8 的 4） |
| throughput | **~330K tok/s**（vs bs=8 的 ~285K） |
| MFU | **~70%**（vs bs=8 的 ~60%） |
| 总训练时间 | 35.4 min (step 5099) |

### 结果（val BPB 变化）

| 时间 | step | val CE | val BPB |
|---|---|---|---|
| 0 min | 0 | 10.06 | 3.2755 |
| ~5 min | 300 | 4.97 | 1.6176 |
| ~10 min | 600 | 4.51 | **1.4687** |
| ~15 min | 900 | 4.27 | 1.3917 |
| **~20 min** | **1200** | 4.16 | **1.3539** |
| ~25 min | 1500 | 4.07 | 1.3264 |
| **~30 min** | **1800** | 4.02 | **1.3107** |
| ~35 min | 2100 | 3.99 | 1.2992 |
| ~40 min | 2400 | 3.95 | 1.2849 |
| ~45 min | 2700 | 3.91 | 1.2748 |
| final (35.4min) | 5099 | 3.58 | **1.1727** |

### 对比

| Round | bs | Tok/s | MFU | 30min BPB | 35min BPB |
|---|---|---|---|---|---|
| 3b | 8 | ~285K | ~60% | **1.3116** | 1.1993 |
| **4** | **16** | **~330K** | **~70%** | **1.3107** | **1.1727** |

**vs bs=8 没有提升！** 30 min 处 BPB 基本持平（1.3107 vs 1.3116）。吞吐从 285K→330K 但没有对应 BPB 改善——模型容量（d12 在 30min/510M token 窗口内）成为瓶颈。35 min 处 bs=16 BPB 1.1727 vs bs=8 1.1993，略有优势但这超出了 30min 窗口。

---

- 目标机器：AutoDL **499机**（25核 Xeon Platinum 8470Q, 2×RTX 5090）
- 策略：vocab=32768, depth=12, seq_len=2048, bs=8 灌满显存
- 启动方式：`torchrun --nproc_per_node=2 -m modalities.text.train_text [参数...]`
- SSH：`wsl ssh -p 17790 root@connect.bjb2.seetacloud.com`
- NANOINFRA_BASE_DIR：`/root/autodl-tmp/nanoinfra/outputs`

## Round 5：2×5090, d12, bs=8, lr=1e-3

```
torchrun --nproc_per_node=2 -m modalities.text.train_text \
  model.depth=12 device_batch_size=8 total_batch_size=131072 \
  optimizer.lr_max=1e-3 max_steps=3600 ...
```

| 参数 | 值 |
|---|---|
| lr_max | **1e-3**（vs baseline 6e-4） |
| throughput | ~285K tok/s |
| MFU | ~60% |
| 总训练时间 | 28.9 min (step 3599) |

### 结果

| 时间 | step | val CE | val BPB |
|---|---|---|---|
| ~5 min | 300 | 5.21 | 1.6968 |
| ~10 min | 600 | 4.63 | 1.5069 |
| ~15 min | 900 | 4.34 | 1.4121 |
| ~20 min | 1200 | 4.19 | 1.3662 |
| ~25 min | 1500 | 4.10 | 1.3348 |
| **~30 min** | **1800** | 4.03 | **1.3138** |
| ~35 min | 2100 | 4.00 | 1.3011 |
| ~55 min | 3300 | 3.75 | 1.2224 |
| final (28.9min) | 3599 | 3.69 | **1.1946** |

**lr=1e-3 跟 lr=6e-4 打成平手！** 30 min BPB: 1.3138 (lr=1e-3) vs 1.3116 (lr=6e-4)。lr 更高初期 loss 掉得快但后期收敛略慢，两个效果互相抵消。

---

## 所有结果汇总

| Round | GPU | Config | Tok/s | MFU | 30min BPB |
|---|---|---|---|---|---|
| 1 | 1×5090 | d10, vocab=32K, bs=2 | ~120K | ~33% | 1.334 |
| 2 | 1×5090 | d10, vocab=64K | ~72K | ~25% | ~1.39 |
| 3a | 2×5090 | d14, bs=2 | ~86K | ~26% | FAIL |
| 3b | 2×5090 | d12, bs=8, lr=6e-4 | ~285K | ~60% | **1.3116** |
| 4 | 2×5090 | d12, bs=16, lr=6e-4 | ~330K | ~70% | **1.3107** |
| 5 | 2×5090 | d12, bs=8, lr=1e-3 | ~285K | ~60% | 1.3138 |
| 7 | 2×5090 | d16, bs=4 | ~140K | ~60% | FAIL（15min BPB=1.47 vs d12 1.35） |
| 8 | 2×5090 | d12, seq=4096, bs=2 | ~190K | ~52% | FAIL（吞吐减半，无改善） |
| 10 | 2×5090 | d14, bs=4 | ~170K | ~52% | FAIL（BPB 1.46 @10min） |

## 结论

1. **最佳配置**: d12, bs=16, lr=6e-4, seq=2048, vocab=32768, 2×5090
   - 30min BPB=**1.3107**，吞吐 ~330K tok/s，MFU ~70%
2. **吞吐是硬约束**——同时间处理更多 token 压倒一切。更深模型（d14/d16）和更长序列（seq=4096）都因吞吐下降而 worse
3. **Batch size 有上限**——bs=8→16 吞吐 +16% 但 30min BPB 不变（1.311→1.310），模型容量成为瓶颈
4. **LR 不敏感**——6e-4 vs 1e-3 持平
5. **大 vocab 是陷阱**——64K vocab 给 BPB 净增 ~0.06

---

## 有用命令存档

```bash
# SSH 连接（从 WSL）
wsl ssh -p 35000 root@connect.bjb1.seetacloud.com

# Python 路径
/root/miniconda3/bin/python3

# 下载 FineWeb shard（需要 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/nanoinfra
/root/miniconda3/bin/python3 exemplars/text_pretrain/data/download_shards.py 006

# 重训 tokenizer
/root/miniconda3/bin/python3 -m modalities.text.train_tokenizer \
  --vocab-size 65536 --force --train-chars 80000000 \
  --out /root/autodl-tmp/nanoinfra/outputs/tokenizer

# 安装依赖
cd /root/autodl-tmp/nanoinfra
/root/miniconda3/bin/python3 -m pip install -e .
/root/miniconda3/bin/python3 -m pip install liger-kernel

# 恢复默认 tokenizer（32768 vocab）
/root/miniconda3/bin/python3 -m modalities.text.train_tokenizer --force
```
