# UPCL

## [NeurIPS2025 Spotlight] Unbiased Prototype Consistency Learning for Multi-Modal and Multi-Task Object Re-Identification
### 🎉🎉🎉 [Paper](<https://arxiv.org/abs/2312.09612](https://proceedings.neurips.cc/paper_files/paper/2025/file/82a0696bea2c4ebf726fc796eaca7a55-Paper-Conference.pdf>)
### Introduction
Unbiased Prototypes-guided Modality Enhancement (UPME) and
Cluster Prototype Consistency Regularization (CPCR). UPME leverages modality-unbiased prototypes to simultaneously enhance cross-modal shared features and
multi-modal fused features. Additionally, CPCR regulates discriminative semantics
learning with category-consistent information through prototypes clustering. Under
the collaborative operation of these two modules, our model can simultaneously
learn robust cross-modal shared feature and multi-modal fused feature spaces,
while also exhibiting strong category-discriminative capabilities.

### Contributions
- To address the practical demands for retrieval of diverse modalities and categories, we propose a
novel Multi-Modal and Multi-Task object ReID (M3T-ReID).
- To address the challenges in M3T-ReID, we propose UPCL which comprises two main components:
UPME and CPCR. UPME leverages modality-unbiased prototypes to simultaneously enhance
cross-modal shared features and multi-modal fused features, and CPCR regulates the learning of
discriminative semantics across diverse object categories through prototypes clustering.
- Extensive experiments on the public multi-modal ReID benchmarks RGBNT201 and RGBNT100
have verified the advantage of our methods, achieving significantly higher accuracy compared to
existing counterparts in both cross-modal and multi-modal retrieval scenarios.

### Framework
<img width="988" height="622" alt="image" src="https://github.com/user-attachments/assets/983efb17-a470-4837-8e92-9aadac186e1a" />

### Results
- Comparison with the state-of-the-art methods. Each model is trained on a combined dataset
consisting of RGBNT201 and RGBNT100, and evaluated separately.
<img width="988" height="423" alt="image" src="https://github.com/user-attachments/assets/3215e8ce-e376-4c95-9652-900fafe21d53" />

- Performance analysis of cross-domain generalization on MSVR310 dataset. All models are
trained on the combined dataset of RGBNT201 and RGBNT100, and evaluated on the MSVR310
dataset.
<img width="988" height="237" alt="image" src="https://github.com/user-attachments/assets/415a80e6-9235-412e-9059-8813db6c448c" />

### Datasets
UNIRNT is constructed by merging the training sets of RGBNT201 and RGBNT100 for unified training. It can be downloaded from link: https://pan.baidu.com/s/19ES_yu5sv4N_e_dWMMfAqg?pwd=cmd8 提取码: cmd8. 

The test sets should be downloaded separately from the original dataset links. During the testing stage, the trained model is evaluated independently on each dataset.
- RGBNT201 link: https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH
- RGBNT100 link: https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug code：rjin
- MSVR310 link: https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view?usp=drive_link


