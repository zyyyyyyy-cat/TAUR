# 实验B（TAUR-v1）完整代码修改指导与最终方案审查

研究任务： WHU-MARS-1000，多模态多平台统一行人重识别。

基础模型： WHU-MARS论文中的UAD，使用ViT-B/16、CE、WRT、ProCA、GPD。

改进目标： 在不改变统一身份检索方式的前提下，提高TIR内部的身份判别能力，并改善TIR与RGB/NIR之间的双向检索表现。

经过最后一轮审查，我建议保留已经确定的TAUR-v1主体，不再新增分类头、原型记忆或蒸馏网络。当前最重要的是实现正确，特别是样本关系构造、梯度更新与推理路径。

下文以[WHU-MARS官方开源工程](https://github.com/msm8976/WHU-MARS/tree/main/CVPR26_UAD)为参考，给出修改位置、模块功能、数据流、损失构造、训练细节和验收标准。这是实现指导，不是已经完成的代码修改。 如果你当前使用的是经过WQVD修改的UAD分支，应以实际文件结构为准，但保持这里定义的行为一致。

## 一、最后一轮审查结论

| 审查角度                     | 结论                               | 最终处理              |
| ------------------------ | -------------------------------- | ----------------- |
| 模块是否针对TIR瓶颈              | 合理：为共享ViT提供TIR特化计算能力             | 保留L6、L9两个Adapter  |
| 是否重复A1最终特征融合             | 不重复：新模块作用于中层Patch                | 最终只使用原CLS         |
| 新增监督是否与UAD重复             | 与WRT有部分重叠，但模态、相机和Gallery条件提供新增约束 | 不再增加其他身份损失        |
| 是否同时改善TIR作为Query和Gallery | 设计覆盖这两类情况                        | 保留四个具体方向          |
| 是否精确优化mAP                | 否，LogSumExp只是多正负排序代理             | mAP和Rank-1共同验收    |
| 训练与推理是否一致                | 只要显式传递modality ID即可实现            | 必须修改训练和测试两个入口     |
| 梯度保护是否合理                 | 合理，但属于特殊参数更新规则                   | 新Loss只更新Adapter   |
| DDP是否需要重写                | 不需要重写UAD训练框架                     | 重点检查每卡有效身份和新增梯度同步 |

结论：目前没有发现必须推翻TAUR-v1的确定性结构矛盾，可以开始实施。 仍然存在的性能风险，主要应交给实际实验判断。

## 二、改进模块的论文来源：必须明确区分“直接借鉴”和“自主改造”

| 参考论文                                                                 | 对应章节或方法                                                                          | 在TAUR中具体采用什么                        | 不采用什么                            |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------- | -------------------------------- |
| [WSL-VIReID](https://github.com/KongLingqi2333/WSL-VIReID)，ICCV 2025 | 第3.2节 Heterogeneous Expert Learning（HEL），图2                                      | 模态特化编码与共享后续骨干的组织思路                  | 独立RGB/IR分类专家、弱监督关系发现与CCL全套流程     |
| [MDReID](https://github.com/stone96123/MDReID)，NeurIPS 2025          | 第3.2节 Modality-Decoupled Learning（MDL）；第3.3节 Modality-Aware Metric Learning（MML） | 模态特有信息不应被一概消除，可以用于增强身份判别            | 专属/共享Token、ROL、KDL、模态Mask及专用检索距离 |
| [MOS](https://github.com/yjzhao1019/MOS)，CVPR 2026                   | 第3.2节 Modality-Consistent Representation Learning（MCRL），类别级模态对齐（CMAL）            | 弱模态增强后还要维持跨模态身份对应的设计原则              | SAR去噪、CMAL均值/方差损失、生成式特征融合        |
| [UPCL](https://github.com/ZhouZhongao/UPCL)，NeurIPS 2025             | 第3.2节 UPME及第3.3节 CPCR                                                            | 作为原型方法对照，避免重复设计模态身份记忆               | 三模态原型库、无偏原型与类别聚类原型损失             |
| [WHU-MARS/UAD](https://github.com/msm8976/WHU-MARS)，CVPR 2026        | 第4.2—4.5节及图3                                                                     | 统一单分支框架、CE/WRT、ProCA/GPD和pre-BN推理契约 | 不修改其原有损失定义和检索协议                  |

论文与代码支持的结论需要准确表述：

WSL-VIReID采用模态专属浅层ResNet和共享后续层；它并没有直接提出我们使用的L6/L9 ViT瓶颈Adapter。

Zhang 等 - Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative C(1).pdf



MDReID明确分离模态共享和模态特有Token，并结合ROL/KDL协调表征；其推理距离组织方式与WHU单向量检索不同，因此TAUR只借鉴信息分工思想。

Feng 等 - MDReID Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification.pdf



MOS研究SAR与光学的类级分布对齐，但UAD已有ProCA，重复增加中心对齐并不是优先选择。

Zhao 等 - 2025 - MOS Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification.pdf



UPCL已经设计模态专属及无偏身份原型，因此TAUR不再建立一套类似原型系统。

Zhou 等 - Unbiased Prototype Consistency Learning for Multi-Modal and Multi-Task Object Re-Identification.pdf



TAUR中的768→128→768瓶颈Adapter、有界残差、条件化多正负LogSumExp排序和梯度路由，是本次针对WHU组合设计的候选实现，不应归属为以上论文中的现成模块。

## 三、完整方法结构：原UAD保留什么、新增什么？

总体结构固定为：

\\[ x\xrightarrow{\text{Shared ViT + TIR Adapter}} f\in\mathbb R^{768} \\]

训练时：

\\[ \mathcal L_U= \mathcal L\_{\mathrm{CE}} +\mathcal L\_{\mathrm{WRT}} +0.01\mathcal L\_{\mathrm{ProCA}} +\mathcal L\_{\mathrm{GPD}} \\]

额外计算：

\\[ \mathcal L\_{\mathrm{rank}} \\]

推理时：

\\[ \hat f=\frac{f}{\\|f\\|\_2} \\]

所有模态都在同一个768维空间中直接比较，不新增查询融合、特征拼接、专用距离函数或重排序。

## 四、需要修改的源码文件和具体职责

根据再次核对的WHU官方目录，主要修改集中在6个功能位置。不建议大范围重构原工程。

| 修改位置                                                                                                                      | 主要改动                             | 是否影响原UAD             |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------------------- |
| [model/backbones/vit_pytorch.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/backbones/vit_pytorch.py) | 增加TIR Adapter，在ViT第6、9个Block后调用  | 改变TIR前向计算            |
| [model/make_model.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/make_model.py)                       | 注册Adapter，向Backbone传递modality ID | 保留原模型输出接口            |
| 新增 `loss/taur_rank_loss.py`                                                                                               | 实现条件化多正负排序损失                     | 不修改原CE/WRT/ProCA/GPD |
| [processor/processor.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/processor/processor.py)                 | 计算新增损失，分配梯度，记录各方向信息              | 保留GPD原更新逻辑           |
| `datasets/bases.py`、`datasets/make_dataloader.py`、`datasets/sampler.py`                                                   | 增加训练样本索引、模态标签及相机优先采样             | 保持PKM分组契约            |
| 配置与优化器构造文件                                                                                                                | 新增TAUR参数，区分Adapter学习率            | 原UAD配置不变             |

建议单独创建TAUR实验分支，保留纯UAD与实验A的现有代码。

### 4.1 Backbone：新增TIR特化Adapter

这是网络层面最主要的改动。

新增模块： `ThermalAdapter`

建议放在独立模块文件中，由ViT调用。

结构固定为：

\\[ U= W\_{\mathrm{up}} \operatorname{GELU} \left( W\_{\mathrm{down}}\operatorname{LN}(H) \right) \\]

其中：

\\[ W\_{\mathrm{down}}:768\rightarrow128 \\]

\\[ W\_{\mathrm{up}}:128\rightarrow768 \\]

对每个样本单独计算有界残差：

\\[ \Delta H= \rho\\, \frac{\\|H\\|\_F} {\\|H\\|\_F+\\|U\\|\_F+\epsilon}\\,U \\]

\\[ H'=H+\Delta H \\]

参数：

| 参数         | 设定               |
| ---------- | ---------------- |
| Adapter数量  | 2                |
| 插入位置       | 第6、9个ViT Block之后 |
| 输入/输出维度    | 768              |
| 瓶颈维度       | 128              |
| 激活         | GELU             |
| 归一化        | LayerNorm        |
| \\(\rho\\) | 0.10             |
| 参数共享       | 两个Adapter不共享     |
| 适用模态       | 仅TIR             |
| 操作Token    | 仅Patch Token     |

必须注意四个实现细节。

首先，ViT的Token序列包含CLS与Patch。若当前使用256×128输入、16×16 Patch及16步长，Patch数量为128，总序列为129。

Adapter只处理Patch部分，CLS必须保持原值并按原顺序放回序列。不能误将129个Token全部送入Adapter。

其次，`torch.nn.ModuleList`中的第6、9个Block对应实际循环索引5、8。应该在执行完对应Block后调用Adapter，而不是循环开始前。

第三，范数约束需要按样本独立计算，不能把整个Batch一起求范数，否则一个样本的残差大小会受到其他样本影响。

第四，Adapter的输出层使用小幅非零初始化，避免整个Adapter完全零初始化后出现不必要的梯度启动问题。原有ViT参数仍从与UAD相同的预训练权重加载。

来源说明： 模态特化与共享编码的设计主要参考WSL-VIReID第3.2节HEL；保留模态特有表征信息的思想参考MDReID第3.2节MDL。这里的中层瓶颈Adapter、有界残差与具体层位均是面向UAD的适配设计，不是直接复制论文中的网络模块。

### 4.2 Model Forward：统一训练和推理的模态处理

这是一个很容易造成整个实验无效的地方。

WHU官方模型训练时先拼接三模态图像，再调用共享ViT；测试时直接传入单Batch图像。

需要让两条前向路径都明确知道每张图像属于什么模态。

建议模型的内部前向统一接收：

- 图像；
- 模态ID；
- 原模型所需的其他元数据。

模态编码沿用数据集已有定义：

| 模态          | ID | Adapter行为        |
| ----------- | -- | ---------------- |
| RGB         | 1  | 不执行              |
| NIR/IR      | 2  | 不执行              |
| TIR/Thermal | 3  | 执行L6、L9两个Adapter |

训练阶段按照RGB、NIR、TIR的拼接顺序生成相同顺序的模态标签。

测试阶段从数据加载器已有的`modid`取得每张图像的真实模态ID，再传入模型。

不要用`model.training`决定是否运行Adapter，也不要把测试时的`mode`变量直接当作模态ID而不核对其含义。

模型返回仍保持原接口：

训练时返回共享分类结果、pre-BN特征和BNNeck特征；推理时只返回当前配置指定的检索特征。

本实验必须使用：

`TEST.NECK_FEAT = before`

也就是原始pre-BN CLS。

还应确认关闭TAUR功能时，原模型能够在相同输入与权重下复现原UAD前向结果。这样可以避免新增接口无意中修改RGB/NIR或基础ViT路径。

## 五、新增排序损失：完整的计算要求

建议新增独立损失模块： `TAURRankingLoss`。

它不负责模型前向，不维护身份原型，只接收特征和元数据，输出新增排序损失及必要统计信息。

### 5.1 损失输入

必须包含：

| 输入            | 用途                 |
| ------------- | ------------------ |
| `global_feat` | 原UAD最终768维pre-BN特征 |
| `pid`         | 判断是否同身份            |
| `camid`       | 排除同相机Gallery       |
| `modality_id` | 区分四种检索方向           |
| `sample_id`   | 排除同一原始图像的重复采样      |

所有数组顺序必须与`global_feat`逐行对应。

这里需要特别注意：WHU训练时不同模态具有独立的身份排列，但在拼接成完整Batch后，不能仍然只使用原来64个样本的`vid`作为排序损失标签。

应与原CE/WRT一样，根据实际拼接顺序得到对应的完整身份标签。

另外，`sample_id`用于标识原始图像，不是训练身份ID。可以使用模态内稳定索引，也可以使用原始图像路径建立标识。

### 5.2 相似度计算

使用：

\\[ \hat f_i=\frac{f_i}{\\|f_i\\|\_2} \\]

\\[ s\_{ij}=\hat f_i^\top\hat f_j \\]

在混合精度训练下，建议将排序相似度与LogSumExp部分转为FP32计算，避免数值不稳定。

不使用BNNeck后的特征，也不使用Adapter输出作为检索特征。

### 5.3 有效正负样本构造

对于Query \\(i\\)，首先构造满足指定Gallery模态条件的候选集合：

\\[ \mathcal G_i^d= \left\\{ j: m_j\in M_d,\\; c_j\ne c_i,\\; \operatorname{sampleID}\_j\ne\operatorname{sampleID}\_i \right\\} \\]

然后按照身份划分：

\\[ \mathcal P_i^d= \\{j\in\mathcal G_i^d:y_j=y_i\\} \\]

\\[ \mathcal N_i^d= \\{j\in\mathcal G_i^d:y_j\ne y_i\\} \\]

最关键的一点：相机过滤必须同时作用于正样本和负样本。

WHU的AS-ReID协议排除的是与Query来自同一相机的全部Gallery图像，而不是只排除同身份同相机样本。

Zhao 等 - WHU-MARS A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification(6).pdf



不能把普通Market-1501的过滤规则直接复制过来。

### 5.4 固定四种训练方向

| 方向    | Query | Gallery正样本  | Gallery负样本  |
| ----- | ----- | ----------- | ----------- |
| T→T   | TIR   | TIR         | TIR         |
| T→ALL | TIR   | RGB/NIR/TIR | RGB/NIR/TIR |
| R→T   | RGB   | TIR         | TIR         |
| N→T   | NIR   | TIR         | TIR         |

这是定稿版本，不能在实现过程中将反向R/N→T的正样本重新扩展为全部模态。

原因是新增排序损失只更新TIR Adapter。如果RGB Query的正样本也是RGB，那么这一关系可能只通过错误TIR负样本向Adapter传递排斥梯度，不能直接增强正确TIR的跨模态匹配。

当前定义明确要求反向组同时包含正确TIR正样本吸引和错误TIR负样本排斥。

### 5.5 正负样本聚合与损失公式

对每个有效正样本 \\(j\\)，构造与所有有效负图像的排序约束：

\\[ \boxed{ \ell\_{ij}^d = \tau\log \left[ 1+\sum\_{k\in\mathcal N_i^d} \exp\left( \frac{s\_{ik}-s\_{ij}+\gamma}{\tau} \right) \right] } \\]

初始配置：

\\[ \gamma=0.03,\qquad\tau=0.07 \\]

这是全负图像LogSumExp，不是Top-K负图像、身份最大负样本或平均Softplus。

实现时应注意以下行为：

- 对每一个有效正样本单独计算，不预先把正样本聚合成一个最高相似度；
- 全部有效负图像均进入LogSumExp；
- 使用稳定数值实现；
- 不因为某个错误身份出现多张图像就把它们全部压缩成一张；
- 不因为负样本数量多就自动对指数和除以当前负样本数，否则将改变已定稿的损失语义。

需要明确它的局限：这个目标保留错误图像数量的影响，但不等价于精确AP。大量容易负样本和不同方向候选数量仍可能造成损失量级差异，因此必须监测各方向的实际梯度贡献。

### 5.6 正样本与Query聚合

先对每个Query的所有有效正样本等权平均：

\\[ L_i^d= \frac1{|\mathcal P_i^d|} \sum\_{j\in\mathcal P_i^d} \ell\_{ij}^d \\]

再对方向内的有效Query平均：

\\[ L_d= \frac1{|\mathcal Q_d|} \sum\_{i\in\mathcal Q_d}L_i^d \\]

首轮不额外增加按相机数、困难程度或图像质量预测的正样本动态权重。

但相机来源和正样本数量应记录，以便判断是否出现少数困难样本长期主导。

如果某方向在某个Batch中没有有效Query，该方向本步贡献记为零，其他方向的权重保持不变。不能因为T→T没有有效样本，就将它的权重自动转给T→ALL。

### 5.7 四方向的最终合成

\\[ \boxed{ L\_{\mathrm{rank}} = 0.25L\_{T\to T} + 0.25L\_{T\to ALL} + 0.25L\_{R\to T} + 0.25L\_{N\to T} } \\]

这也可以解释为：

TIR Query两方向合计权重0.5，TIR Gallery两方向合计权重0.5。

但它只是损失系数分配，不代表两类任务各自贡献50%的实际梯度。

T→T与T→ALL还包含部分重复关系，这是有意强调TIR内部身份判别与混合Gallery检索，并非完全独立的两项监督。

来源说明： 条件正负样本的构造直接依据WHU的AS-ReID定义；多模态双向身份监督受到WSL-VIReID与UPCL相关思想启发。但这套具体的四方向、全图像LogSumExp排序不是上述论文的原有损失，应作为TAUR候选设计独立描述。

## 六、训练器：原UAD损失完全保留，新增损失只更新Adapter

这一部分是实现中最需要严谨处理的地方。

### 6.1 参数分组

建议将可训练参数分成两组。

共享参数 \\(\theta\\)： 原ViT、BNNeck、共享分类器，以及UAD其他原有可训练参数。

TIR特化参数 \\(\phi\\)： 两个新增Adapter的参数。

原UAD全部损失继续作用于共享参数和Adapter。

新增排序损失只作用于Adapter。

因此实际更新为：

\\[ \boxed{ g\_\theta=\nabla\_\theta L_U } \\]

\\[ \boxed{ g\_\phi= \nabla\_\phi L_U+ \lambda\nabla\_\phi L\_{\mathrm{rank}} } \\]

这一点非常重要。

不能简单计算`L_U + λL_rank`，然后对整个网络执行普通反向传播。

那样新增排序损失也会直接更新共享ViT，违背当前的梯度路由设计。

### 6.2 正确的训练器职责

训练器应在一次正常前向中得到统一CLS，随后分别计算UAD损失和新增排序损失。

进行梯度处理时：

- UAD损失按原方式向全部可训练参数反向传播；
- 排序损失只求相对于两个Adapter参数的梯度；
- Adapter的两部分梯度合并；
- 共享参数不接收排序损失的直接梯度；
- 最后统一执行梯度裁剪与优化器更新。

不得把整个共享Transformer放进`no_grad()`。

因为排序损失需要通过后续Transformer的计算关系，把最终CLS上的误差信号传回中层Adapter。如果切断后续计算图，Adapter就收不到正确梯度。

同时，不能在新增排序损失之前将TIR最终特征整体`detach()`，否则新增监督也无法更新Adapter。

这属于一种参数分组的梯度路由策略，应在论文方法与训练实现中明确说明，而不是作为普通联合Loss的隐含处理。

### 6.3 与原UAD的四种损失如何相处？

| 原UAD损失    | 是否保留 | 与新增方法的关系                    |
| --------- | ---- | --------------------------- |
| CE        | 完全保留 | 继续监督共享身份分类                  |
| WRT       | 完全保留 | 保留混合Batch度量学习，新Loss强调条件检索排序 |
| ProCA     | 完全保留 | 继续约束不同模态的同身份中心              |
| GPD       | 完全保留 | 继续提供训练集级别的全局身份原型判别          |
| TAUR Rank | 新增   | 只通过Adapter加强TIR相关条件排序       |

不修改原GPD的身份原型更新，不增加第二套模态身份原型，也不额外增加TIR CE或正交损失。

### 6.4 学习率与损失启用

初始配置建议：

| 参数           | 建议                          |
| ------------ | --------------------------- |
| 总训练Epoch     | 120                         |
| 原Backbone学习率 | 与当前UAD基线完全一致                |
| Adapter学习率   | 原Backbone的3倍                |
| 原UAD损失系数     | 全部不变                        |
| 排序损失最终权重     | 0.10                        |
| Epoch 1—5    | 排序损失权重为0，Adapter仍参与UAD训练    |
| Epoch 6—20   | 排序损失权重线性增加至0.10             |
| Epoch 21—120 | 排序损失保持0.10                  |
| 优化器及调度       | 延用当前UAD的SGD、Warmup、Cosine设置 |

这些是首轮固定实验参数，不是已经通过WHU验证的最优参数。

特别注意：虽然Epoch 1—5排序损失为0，但Adapter本身不能关闭。它仍应通过原UAD训练，以免Epoch 6突然改变前向路径。

另外，最终优化预算应以你当前真正使用的UAD运行配置为准。官方公开配置的`LOSS_TYPE`默认可能是`base`，必须确认本轮实际上启用了ProCA和GPD；否则训练出来的就不是“UAD+TAUR”。

## 七、数据加载与PKM采样：必要修改，但不改变基本协议

当前PKM的逻辑仍保留：

\\[ P=16,\qquad M=3,\qquad K=4 \\]

理论完整Batch为：

\\[ 16\times3\times4=192 \\]

原采样器按身份和模态采样，但没有保证同一身份的TIR样本来自不同相机。

而新增Loss需要跨相机有效正样本。

因此建议采用相机优先的PKM采样：

对每个训练身份及每个模态，优先选择来自不同相机的样本；若只有一个相机，则维持原始随机采样；剩余位置按原PKM方式补足。

需要特别明确：

1. 保持每身份每模态K=4及原模态顺序。
2. 不改变ProCA按每组4张图像计算模态中心的分组结构。
3. Ground/Aerial不能代替真实camera ID。
4. 不应为了满足跨相机监督而把同相机样本人为标记为不同相机。
5. 不改变训练集身份标签、训练/测试划分及官方Gallery过滤规则。

如果某身份的TIR确实只在一个相机中出现，它仍接受UAD监督，但不能凭空构造严格的TIR→TIR跨相机正样本。

此外，原PKM曾存在`__len__`与实际产出数量不一致的问题。务必以实际Optimizer Step和有效样本数比较训练预算，不要仅用“120 Epoch”判断两个实验是否完全同等。

由于采样器改动可能单独影响性能，正式消融应包含使用相同采样策略的UAD对照。

## 八、DDP：只在原UAD基础上增加必要检查

这里不建议重写DDP框架，也不引入新的全局身份记忆。

第一，保持原UAD的模型同步和GPD原型更新方式。 不改变其原有身份中心聚合与动量原型流程。

第二，检查每卡实际用于排序的有效身份和正样本。 全局192张不等于每卡都拥有192张。如果本地候选只有少数身份，新增Loss的困难负样本覆盖可能偏弱；跨相机正样本若分散在不同卡上，也可能无法构成有效关系。

第三，只有本地候选覆盖明显不足时，才需要引入跨卡特征聚合。 如果确实使用跨卡特征，必须保证需要接收排序梯度的TIR特征仍可反向传播，不能把所有远端特征都无条件`detach()`后声称等价于全局排序训练。

第四，新增Adapter梯度必须正确同步。 原UAD的DDP同步并不意味着通过单独求导获得的排序梯度必然自动同步。按现有AMP与DDP训练路径正确合并、同步和裁剪即可，无需重写优化器体系。

建议先用单卡小Batch检查新增损失与梯度，再用实际多卡配置检查各方向有效Query数量和梯度是否正常。这里属于代码正确性检查，不需要额外训练一套诊断模型。

## 九、推理阶段：只增加TIR路由，其他协议不变

推理流程固定为：

推理不需要身份标签，不使用新排序损失，不使用任何训练期原型扩展，也不重新生成特征融合向量。

必须保证： 同一个Checkpoint、同一个TIR输入，在训练路径的评估状态与正式测试路径下，得到的pre-BN特征一致（允许正常浮点误差）。

Adapter不能只在训练时启用；测试的Query和Gallery中只要是TIR，都必须经过对应Adapter。
