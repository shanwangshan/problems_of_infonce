## Underlying defect of InfoNCE loss

1. InfoNCE loss is widely used as a contrastive loss during the self-supervised learning process. However, studies have found out that, with two different embedding distributions on a hypersphere, the contrastive loss may end up with the same value even though one distribution may be more useful for the downstream tasks.
2. Studies also have shown that this loss is less tolerant to semantically similar samples due to the inherent defect of the instance discrimination objectives, which may harm the quality of the learnt feature embeddings.

In this git repo, we provide a simple example in two dimensional space to illustrate this. To run this code, run the command below,

`` python infonce.py ``

As shown in the Fig ![illustration](/Infonce_problem.pdf), there are two different distributions (a) and (b) of piano and rain feature representations. If *piano aug2* is switched with *rain aug2*, the contrastive loss still remains the same. However, it is apparent that distribution (a) is more semantically meaningful for the downstream tasks because it groups samples from the same class closer to each other.

To solve the problem mentioned above, we propose to use angular contrastive loss (ACL) where the details are seen from [infonce](https://github.com/shanwangshan/problems_of_infonce).
