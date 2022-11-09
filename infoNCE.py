import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


import numpy as np


origin = [0,0]

piano_aug1_emb = [1,0.5]
piano_aug2_emb = [1,0]

rain_aug1_emb = [1,1]
rain_aug2_emb = [0,1]

# exchange the dog_barking aug2 with the water_dropping aug2
piano_aug2_emb_ex = [0,1]
rain_aug2_emb_ex = [1,0]


CEL = torch.nn.CrossEntropyLoss()

def info_nce_loss(features):
    labels = torch.cat([torch.arange(2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)


    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / 0.07
    loss = CEL(logits,labels)

    return logits, labels,loss



features1 = torch.tensor([piano_aug1_emb,rain_aug1_emb,piano_aug2_emb,rain_aug2_emb])
features2 = torch.tensor([piano_aug1_emb,rain_aug1_emb,piano_aug2_emb_ex,rain_aug2_emb_ex])

_,_,loss1 = info_nce_loss(features1)
_,_,loss2 = info_nce_loss(features2)

print('loss1 is', loss1,'and loss 2 is',loss2)

plt.rcParams.update({'font.size': 20})

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot([origin[0], piano_aug1_emb[0]],[origin[1], piano_aug1_emb[1]], 'b',linewidth=4, linestyle="--",label='piano aug1 [1,0.5]')
ax1.plot([origin[0], piano_aug2_emb[0]],[origin[1], piano_aug2_emb[1]], 'b',linewidth=4, linestyle="-",label='piano aug2 [1,0]')
ax1.plot([origin[0], rain_aug1_emb[0]],[origin[1], rain_aug1_emb[1]], 'r',linewidth=4, linestyle="--",label='rain aug1 [1,1]')
ax1.plot([origin[0], rain_aug2_emb[0]],[origin[1], rain_aug2_emb[1]], 'r', linewidth=4,linestyle="-",label='rain aug2 [0,1]')
ax1.legend()

ax2.plot([origin[0], piano_aug1_emb[0]],[origin[1], piano_aug1_emb[1]], 'b',linewidth=4, linestyle="--",label='piano aug1 [1,0.5]')
ax2.plot([origin[0], piano_aug2_emb_ex[0]],[origin[1], piano_aug2_emb_ex[1]], 'b',linewidth=4, linestyle="-",label='piano aug2 [0,1]')
ax2.plot([origin[0], rain_aug1_emb[0]],[origin[1], rain_aug1_emb[1]], 'r',linewidth=4, linestyle="--",label='rain aug1 [1,1]')
ax2.plot([origin[0], rain_aug2_emb_ex[0]],[origin[1], rain_aug2_emb_ex[1]], 'r',linewidth=4, linestyle="-",label='rain aug2 [1,0]')
ax2.legend()


plt.show()
