## Underlying defect of InfoNCE loss

### Background

1. InfoNCE loss is widely used as a contrastive loss during the self-supervised learning process. However, study, such as ([Understanding the Behaviour of Contrastive Loss](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf), has found out that, with two different embedding distributions on a hypersphere, the contrastive loss may end up with the same value even though one distribution may be more useful for the downstream tasks.
2. It also pointed out that this loss is less tolerant to semantically similar samples due to the inherent defect of the instance discrimination objectives, which may harm the quality of the learnt feature embeddings.

### Problem illustration

In this git repo, we provide a simple example in two dimensional space to illustrate this. To run this code, run the command below,

`` python infonce.py ``

### Results analysis

As shown in the Fig below, ![illustration](/Figure_1.png), there are two different distributions (a) and (b) of piano and rain feature representations. If *piano aug2* is switched with *rain aug2*, the contrastive loss still remains the same. However, it is apparent that distribution (a) is more semantically meaningful for the downstream tasks because it groups samples from the same class closer to each other.

### Solution
To solve the problem mentioned above, we propose to use angular contrastive loss (ACL) where the details are seen from [supervised_ACL](https://github.com/shanwangshan/supervised_ACL), [self-supervised_ACL](https://github.com/shanwangshan/Self_supervised_ACL), and [feature analysis](https://github.com/shanwangshan/uniformity_tolerance). For more details, please see our paper [*"Self-supervised learning of audio representations using angular contrastive loss"*](https://arxiv.org/abs/2211.05442).
