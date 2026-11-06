# TAUR 的参考论文与本地代码核对

依据为用户提供的五篇本地 PDF、`idea.md` 和相应仓库的模型、loss、训练、数据及评测路径。关键公式页已提取并渲染核对。采用顺序为论文定义优先，代码用于确认具体路径，发现不一致不机械照抄。

## WHU-MARS / UAD

- 论文第 3.3/4.1 节及式 (2)：AS-ReID 排除 Query 所在相机的**所有** Gallery 图像，包括不同身份。这与本地 `utils/metrics.py` 的 `g_camids == q_camid` 过滤一致，TAUR 正负候选都遵守。
- 第 4.2 节：共享 backbone、BNNeck、classifier；CE 对 BN 特征、WRT 对 pre-BN 特征；测试归一化 pre-BN。
- 第 4.3 节式 (4)—(6)：先聚合同模态多视角，再聚合模态中心形成 stop-gradient 身份中心。对应 `processor.PCA` 的连续 K 分组，因此新 sampler 必须完整保留身份组。
- 第 4.4 节式 (7)—(9)：每步先用固定全局原型计算身份判别，再更新 memory。原中心归一化与更新逻辑保持原样。
- 第 5.1 节：ViT-B、SGD、120 Epoch、warm-up 5、ProCA=0.01/GPD=1、GPD 温度 0.03、旧原型权重 0.8。与本地 `configs/UAD.yml`/processor 核对一致。批量和 LR 的“每卡”与本地全局配置语义差别在实现说明中单独记录。
- 本地 train/test 入口已有单卡、torchrun DDP、分模态验证、DataParallel 推理、36 域协议等功能；全部保留。未找到 WHU 实验日志和训练权重，不作基于成绩的推断。

## MDReID

- 第 3.2 节 MDL：在共享 ViT 中加入 modality-specific 和 modality-shared Token；使用六组件表示与 availability mask，并分别计算同模态特有信息与跨模态共享信息的检索相似度。
- 第 3.3 节 MML：ROL 对六种表示建立相似度目标，specific 分量彼此与 shared 分量正交，shared 分量互相接近；KDL 用组合/单分量正负距离比值促进互补。
- 本地 `modeling/make_model.py`、backbone、`engine/processor.py` 的 CoRefine/orthonomal_loss、`layers/triplet_loss.py` 和检索工具对应这些路径，不能直接塞进 WHU 单 CLS 检索契约。
- 论文 KDL 附近文字中的正负距离大小号，与紧接的式 (15) 和“正样本距离应小、负样本距离应大”的目标描述存在相反方向。TAUR 不采用 KDL，避免为无关复现细节改变实验主体。
- 本次只借鉴“模态特有信息仍可用于身份判别”；没有实现双 Token、ROL、KDL、availability mask 或特有检索距离。因此不能称 TAUR 为 MDL 的直接复现。

## WSL-VIReID

- 第 3.1 节是模态内部身份已知、跨模态身份对应未知的弱监督问题，和 WHU 的全监督身份数据不同。
- 第 3.2 节 HEL：RGB/IR 模态专属 ResNet 浅层、共享后续层，独立身份分类专家及模态内部 CE/WRT。
- 第 3.3 节 CRE：专家跨模态预测形成一致、单向及冲突对应关系；第 3.4 节 CCL：对应关系引导共享分类与跨模态 WRT、专家一致性学习。
- 本地 `WSL_ReID/models/agw.py` 明确 `rgb_layers/ir_layers/common_layers`；`task/train.py`、`wsl.py` 使用对应关系矩阵与模态 memory/分类专家训练，支持对 HEL 组织方式的理解。
- 本次借鉴模态特化计算与共享后续编码。不增加独立分类专家、伪关系预测、弱监督 CCL 或 memory；L6/L9 ViT 瓶颈 Adapter 是自主适配，WSL 没有提出这些具体层位或结构。

## MOS

- 第 3.2 节 MCRL：针对 SAR speckle 的低分位像素处理；CMAL 对每个同时出现于 Optical/SAR 的身份建立类别条件统计匹配。
- 论文式 (6)—(7) 从 Gaussian Wasserstein 距离解释均值与标准差，但最终式 (12) 采用均值和**方差**向量差的平方 L2 和。这里不能把方差与标准差混称。
- 第 3.3 节 CDGF：Brownian Bridge diffusion 生成伪 SAR，测试将真实与生成图像特征按权重融合。目标任务、成像噪声及推理方式均不同于 WHU 单向量流程。
- 本地 `loss/make_loss.py` 的 classwise 函数对特征先归一化，再使用均值和方差的**未平方 L2 范数**，不同于论文式 (12) 的平方范数。代码另有 whole-batch 对齐函数，不应误当作类别条件 CMAL。
- 本次只借鉴增强弱模态后仍需维持跨模态身份关系的原则。UAD 已有 ProCA，按方案不叠加 CMAL；不引入 SAR 预处理、生成模型或推理融合，因此上述代码/论文差异不传播到 TAUR。

## UPCL

- 第 3.1 节：共享 CLIP 编码、CE/triplet 和双向跨模态损失；任务同时包含跨模态、多模态融合与人/车辆多类别。
- 第 3.2 节 UPME：模态身份原型按三模态平均形成 unbiased prototype，再用原型身份信息增强三种模态。
- 第 3.3 节 CPCR：分别聚类人/车辆 unbiased prototype，簇内平均后再按簇等权平均得到类别原型，进行类别判别；这是类别级语义约束，WHU 单行人任务没有同样的类别区分需求。
- 本地 `modeling/make_model.py`、`engine/processor.py`、`utils/cluster.py`、`utils/finch.py` 对应模态 memory、unbiased memory 和 FINCH；公开代码 CPCR 路径使用 feature-to-category-prototype MSE，和论文类别 softmax 比率式 (16)—(18) 并不相同。论文该处缺少通常负 log 的负号，不能只靠照搬文字公式确定最小化方向。
- 本次将其作为原型方法对照，保留 UAD 已有 GPD，不新增 UPME/CPCR、模态身份原型或类别簇 memory。不能称四方向 TAUR 排序为 UPCL 原有损失。

## 方案审查后的选择

保留 TAUR 的候选结构和四方向损失，不因参考方法更复杂而扩展成多头、多 Token、多原型或蒸馏模型。补充的实现性修正集中在：明确 sample ID/真实 modid、FP32 排序、checkpoint/config 一致性、持续 3 倍 Adapter schedule、DDP 梯度与 BN buffer 同步、准确 sampler 长度及相同采样的消融对照。

这些修正用于使实验定义可执行且可审查，没有保证性能提升。参数 rho=0.10、瓶颈 128、L6/L9、margin=0.03、temperature=0.07、λ=0.10 都仍是首轮候选设置。

实现同步路线另参考 PyTorch 官方说明：[DDP 的 autograd.grad 限制与 no_sync](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)、[AMP 示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)。按 no_sync 延后归约、合并 scaled 梯度后只 unscale 一次，避免把新梯度当作 DDP 自动同步结果。
