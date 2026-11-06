# 运行日志审查后的修正说明

日期：2026-10-12。本轮针对运行到 Epoch 16 的日志审查结果修正，不改变两个 Adapter 的结构、四方向排序公式、梯度路由、ProCA/GPD 定义和 WHU 评测协议。

## 修正内容

### 1. AMP 梯度诊断与真实更新记录

`processor/taur_grad.py` 新增诊断工具：先对 GradScaler 缩放的损失求导，再将返回的梯度以 FP32 除回 scale。诊断不修改 `.grad`，不会参与参数更新。此顺序依据 [PyTorch 2.2 AMP 示例](https://docs.pytorch.org/docs/2.2/notes/amp_examples.html#gradient-penalty)。

各方向记录 `0.25λL_d` 的本卡梯度范数和跨卡平均梯度向量的范数；后者不是本卡范数的算术平均。额外记录完整 `λL_rank` 与 UAD 的 Adapter 梯度范数、二者比值和余弦，均在梯度裁剪前计算。DDP 诊断的临时向量另做归约，不改动训练梯度；训练梯度仍只归约一次。

训练日志明确区分 UAD Loss、Rank Loss、weighted Rank、λ、全模型裁剪前梯度范数、AMP scale。根据标准单优化器 GradScaler 的公开 scale 变化统计成功更新和 overflow 跳步：scale 回退表示跳步。出现跳步会记录警告。不依赖 GradScaler 私有字段。GPD 的原每步更新方式保留。

### 2. 相机优先 PKM：消费本轮池，随机化批次

保留原 max-modality-count/K 的身份预算，以及准确长度、整 K 分片和各 rank 相同步数。

每身份每模态先建立本轮图像池：图像充足时不重复；图像不足时按完整池重复和随机余量补足，和原 PKM 的补样原则一致。按相机优先分组时实际消费池内图像；后续不从完整源图像池反复重抽。尽量避免组内重复，实在不足才允许重复，sample ID 仍用于排除同原图关系。

排程只强制选择不立即安排就无法在剩余步数完成预算的身份，其余身份随机选择；再打乱完整批次、模态分组和批内身份顺序。取消“每次优先剩余组数最多”的轮次顺序。所有随机操作都来自 `Random(seed+epoch)`，各 rank 生成相同全局计划。

相机数量极不均衡时，少数相机可能在本轮图像池中耗尽。此时维持剩余图像的真实相机 ID，不重复少数相机图像以制造跨相机监督。因此修正后的有效 T2T Query 比例不一定再是 100%，这是覆盖率与真实关系约束共同作用的结果，不应为恢复 7488/7488 而伪造或重复样本。

日志增加全局抽取次数、各模态实际唯一图像数、实际身份数和步数。合成 16 身份×100 图像/模态的检查由旧实现 1600 次抽取仅覆盖 1011 张 RGB 原图，改善为抽取/唯一图像均为 1600。

### 3. 权重加载：完整性校验与原子拒绝

`model/make_model.py` 支持原始 state_dict、state_dict/model 容器和 module 前缀；检查共享骨干、BN 状态、完整 Adapter 键及形状，拒绝未知必要键和缺失参数。所有检查在复制参数前完成，加载失败不会留下部分已更新的模型。

测试时外层 classifier 和未使用的 base.fc 不影响检索，可以省略；必要的共享特征提取器和 Adapter 必须完整。分类头不因训练身份数改变而阻止检索加载。

`load_param_finetune` 明确允许完整 UAD checkpoint 初始化 TAUR 共享参数并保留 Adapter 初始值；部分 Adapter checkpoint 仍拒绝。它是权重初始化，不是恢复 optimizer、scheduler、AMP scaler 或 GPD memory 的完整续训。

### 4. WRT 的数值修正与公平对照

`MODEL.WRT_STABLE=True` 使用有效样本内的稳定 masked softmax，修复较大负距离下权重和远小于 1 的问题。空集合输出可微的零，非空集合权重和为 1，保持原 WRT 的正负加权距离与 SoftMarginLoss 结构。

原 `configs/UAD.yml` 未修改，`WRT_STABLE` 默认 False，原数值行为仍可复现。**新版 `TAUR.yml` 启用 True**；新增 `configs/UAD_TAUR_control.yml` 同样启用稳定 WRT 和修正后的相机优先采样，但关闭 Adapter/排序。比较 TAUR 的模块增益应使用此匹配对照，不能将新版与旧日志的变化全部归因于 TAUR。

若仅希望比较诊断/采样修正、继续沿用旧 WRT，可在两组命令中同时覆盖 `MODEL.WRT_STABLE False`。不能只在其中一组改变这个开关。

### 5. 自检兼容性与真实入口检查

新增 `tests/amp_compat.py`：优先使用可用的 native CPU GradScaler；旧 Torch 没有该接口时使用仅供 CPU 测试的固定尺度/overflow 检查类，避免测试直接依赖 Torch 2.8 的接口。这个测试替身不用于训练或替代 CUDA AMP。

训练器/推理器现尊重已有 `MODEL.DEVICE=cpu` 设置，使真实入口可在无 GPU、无数据集时接受合成数据检查。默认仍为 cuda，DDP 训练入口仍使用原 torchrun/NCCL 初始化；原 GPU AMP API 保留以兼容 Torch 2.2.2。

新版配置每 10 Epoch 保存一次原格式模型权重，并使用新的输出目录，减少丢失整段训练模型权重的风险；不宣称新增完整续训功能。

## 本轮文件清单

修改 11 个文件，新增 4 个文件，无删除。

| 文件 | 变更 |
|---|---|
| `processor/taur_grad.py` | AMP 安全诊断、临时梯度向量、局部/平均贡献、更新结果统计 |
| `processor/processor.py` | 使用新诊断、实际更新/跳步、损失与采样覆盖日志；CPU 入口检查支持 |
| `datasets/sampler_camera.py` | 消费本轮图像池、随机可行排程、覆盖统计 |
| `model/make_model.py` | 必要权重完整性和形状检查、无部分加载、容器/前缀支持 |
| `loss/wrt_loss.py` | 可选稳定 masked softmax；保留 legacy 分支 |
| `loss/make_loss.py` | 接入 WRT 开关；CPU 时不创建 CUDA center criterion |
| `config/defaults.py` | 增加 WRT_STABLE，默认 False |
| `configs/TAUR.yml` | 启用稳定 WRT；每 10 Epoch 保存；输出目录 v2 |
| `tests/test_taur.py` | 使用 CPU scaler 兼容工具；调整相机覆盖验收以符合消耗池策略 |
| `tests/check_taur_ddp.py` | 兼容 CPU scaler；新增方向诊断、单 rank overflow 全 rank 跳步检查 |
| `docs/TAUR_IMPLEMENTATION.md` | 更新最终配置、采样行为、验证证据及文件清单 |
| 新增 `configs/UAD_TAUR_control.yml` | 相同稳定 WRT、相同采样的 UAD 对照 |
| 新增 `tests/amp_compat.py` | Torch 2.2 CPU 测试兼容工具 |
| 新增 `tests/test_taur_fixes.py` | 审查修正的回归与真实入口合成检查 |
| 新增 `docs/TAUR_FIXES.md` | 本修正说明 |

修正前对应文件备份位于根目录 `tmp/taur_review_original/`。

## 自检证据与限制

- `python -m pytest -q tests`：**31 passed**。包括原 12 项检查及诊断下溢、图像覆盖、随机预算、四 rank 分片、不足模态补样、权重错误原子拒绝、稳定 WRT 值/梯度、原配置和匹配对照等检查。
- 真实 `do_train` 入口在 CPU 合成数据上训练 6 Epoch，覆盖 λ=0 到 λ>0 的切换、真实 CE/WRT/ProCA/GPD、优化器/调度器、方向诊断、实际更新日志、验证及 `do_inference`；无需真实数据或权重。
- 语法编译与全部 WHU Python 文件的 Python 3.8 语法解析通过。
- 模拟旧 Torch 缺少 CPU GradScaler 的命名空间，兼容回退及 overflow/跳步检查通过。没有安装并执行整个 Torch 2.2.2 环境。
- 本机 PyTorch 2.8.0+cpu 的 Gloo 初始化仍失败，显式尝试 TCP transport 也报 unsupported device。因此增强后的真实双进程同步检查**仍未通过本地运行验证**，必须在 Linux 服务器执行；模拟分片与归约检查不能代替真实 DDP。
- 无 CUDA/NCCL，未验证真实 GPU FP16、四卡性能、真实权重贡献及最终 mAP/Rank-1。

## 新实验命令

在 WHU 目录执行，路径按服务器实际情况调整：

```bash
# 修正版 TAUR，保持上一轮四卡 batch/LR
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/TAUR.yml MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  TAUR.LOG_DIRECTION_GRADS True OUTPUT_DIR logs/TAUR_1000_DDP_v2

# 使用相同采样和稳定 WRT 的 UAD 对照
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/UAD_TAUR_control.yml MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  OUTPUT_DIR logs/UAD_TAUR_control_DDP_v2

# 测试：不要再次读取 ImageNet 权重
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/TAUR.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_CHOICE none \
  TEST.WEIGHT logs/TAUR_1000_DDP_v2/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True OUTPUT_DIR logs/TAUR_1000_DDP_v2_eval

python -m pytest -q tests
python tests/check_taur_ddp.py
```

原有 ROOT_DIR 和 PRETRAIN_PATH 默认值保留。新版会改变采样与 WRT 数值过程，正式对照建议从相同 ImageNet 初始化重新训练，并使用新输出目录。已运行进程不会因为本地文件修正而自动应用新代码。旧 checkpoint 的模型结构未改变，仍可使用完整性检查后的测试加载。
