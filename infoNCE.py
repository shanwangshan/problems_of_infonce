import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def info_nce_loss(features):

    labels = torch.cat([torch.arange(4) for i in range(2)], dim=0)
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
    # using torch CEL, there are two ways of doing it
    #loss1 = CEL(similarity_matrix/0.07,labels)

    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    loss = CEL(logits,labels)

    #import pdb; pdb.set_trace()
    return logits, labels,loss


CEL = torch.nn.CrossEntropyLoss()

origin = [0,0]
p11 = [1.0,0.0] # piano augmented version 1
p12 = [1.0,0.2] # piano augmented version 2
v11 = [1.0,0.3] # violin augmented version 1
v12 = [1.0,0.4] # violin augmented version 2


r11 = [0.0,1.0] # rain augmented version 1
r12 = [0.2,1.0] # rain augmented version 2
s11 = [0.3,1.0] # snow augmented version 1
s12 = [0.4,1.0] # snow augmented version 2

p11_ex = [0,1] # piano 1 switched with rain 1
p12_ex = [0.2,1] # piano 2 switched with rain 2
r11_ex = [1.0,0.0] # rain 1 switched with piano 1
r12_ex = [1.0,0.2] # rain 2 switched with piano 2


plt.rcParams.update({'font.size': 20})
fig, (ax1, ax2) = plt.subplots(1, 2)


ax1.plot([origin[0], p11[0]],[origin[1], p11[1]], 'b',linewidth=4, linestyle="--",label='p11 [1,0]')
ax1.plot([origin[0], p12[0]],[origin[1], p12[1]], 'b',linewidth=4, linestyle="-",label='p12 [1,0.2]')
ax1.plot([origin[0], v11[0]],[origin[1], v11[1]], 'g',linewidth=4, linestyle="--",label='v11 [1,0.3]')
ax1.plot([origin[0], v12[0]],[origin[1], v12[1]], 'g',linewidth=4, linestyle="-",label='v12 [1,0.4]')
ax1.plot([origin[0], r11[0]],[origin[1], r11[1]], 'r',linewidth=4, linestyle="--",label='r11 [0,1]')
ax1.plot([origin[0], r12[0]],[origin[1], r12[1]], 'r',linewidth=4, linestyle="-",label='r12 [0.2,1]')
ax1.plot([origin[0], s11[0]],[origin[1], s11[1]], 'y',linewidth=4, linestyle="--",label='s11 [0.3,1]')
ax1.plot([origin[0], s12[0]],[origin[1], s12[1]], 'y',linewidth=4, linestyle="-",label='s12 [0.4,1]')
ax1.legend()

ax2.plot([origin[0], p11_ex[0]],[origin[1], p11_ex[1]], 'b',linewidth=4, linestyle="--",label='p11_ex [0,1]')
ax2.plot([origin[0], p12_ex[0]],[origin[1], p12_ex[1]], 'b',linewidth=4, linestyle="-",label='p12_ex [0.2,1]')
ax2.plot([origin[0], v11[0]],[origin[1], v11[1]], 'g',linewidth=4, linestyle="--",label='v11 [1,0.3]')
ax2.plot([origin[0], v12[0]],[origin[1], v12[1]], 'g',linewidth=4, linestyle="-",label='v12 [1,0.4]')
ax2.plot([origin[0], r11_ex[0]],[origin[1], r11_ex[1]], 'r',linewidth=4, linestyle="--",label='r11_ex [1,0]')
ax2.plot([origin[0], r12_ex[0]],[origin[1], r12_ex[1]], 'r',linewidth=4, linestyle="-",label='r12_ex [1,0.2]')
ax2.plot([origin[0], s11[0]],[origin[1], s11[1]], 'y',linewidth=4, linestyle="--",label='s11 [0.3,1]')
ax2.plot([origin[0], s12[0]],[origin[1], s12[1]], 'y',linewidth=4, linestyle="-",label='s12 [0.4,1]')
ax2.legend()
plt.show()

features0 = torch.tensor([p11,v11,r11,s11,p12,v12,r12,s12])
features1 = torch.tensor([p11_ex,v11,r11_ex,s11,p12_ex,v12,r12_ex,s12])
_,_,loss0 = info_nce_loss(features0)
_,_,loss1 = info_nce_loss(features1)

print(loss0,loss1)
#import pdb; pdb.set_trace()
