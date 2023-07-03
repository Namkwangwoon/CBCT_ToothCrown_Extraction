import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=2, gamma=4, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        # prob = torch.sigmoid(output)
        prob = output
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
<<<<<<< HEAD
        return loss

TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18',
             '21', '22', '23', '24', '25', '26', '27', '28',
             '31', '32', '33', '34', '35', '36', '37', '38',
             '41', '42', '43', '44', '45', '46', '47', '48']
GD_PAIR = {
    18 : 17,
    17 : 16,
    16 : 15,
    15 : 14,
    14 : 13,
    13 : 12,
    12 : 11,
    11 : 21,
    21 : 22,
    22 : 23,
    23 : 24,
    24 : 25,
    25 : 26,
    26 : 27,
    27 : 28,
    48 : 47,
    47 : 46,
    46 : 45,
    45 : 44,
    44 : 43,
    43 : 42,
    42 : 41,
    41 : 31,
    31 : 32,
    32 : 33,
    33 : 34,
    34 : 35,
    35 : 36,
    36 : 37,
    37 : 38,
}

def find_heatmap_channel(teeth_idx, YES_TEETHS):
    h_cnt = 0
    real_idx = TOOTH_NUM.index(str(teeth_idx))
    for h in range(real_idx):
        if YES_TEETHS[h] == 0:
            continue
        else:
            h_cnt += 1
    return h_cnt

class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()
        
    def forward(self, heatmaps, removes):
        
        '''
        removes.shape = [1, 25, 32]
        '''
        losses = []
        removes = removes.squeeze(0)
        YES_TEETHS = torch.sum(removes, dim=0)
        '''
        '''
        # print('TOOTH_NUM : ',TOOTH_NUM)
        # print('YES_TEETHS : ',YES_TEETHS)
        YES_TEETHS = YES_TEETHS.tolist()
        # YES_TEETHS = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # YES_TEETHS = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
        # YES_TEETHS = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        '''
        0 -> 12
        1 -> 15
        2 -> 17
        3 -> 18
        4 -> 19
        '''
        for idx in range(16):
            # heatmap_cur_idx = YES_TEETHS[idx]
            # print('heatmap_cur_idx : ', heatmap_cur_idx)
            if YES_TEETHS[idx] == 0:
                continue
            teeth_cur_idx = TOOTH_NUM[idx]
            if teeth_cur_idx in ['28', '38']:
                continue
            teeth_near_idx = GD_PAIR[int(teeth_cur_idx)]
            # heatmap_near_idx = YES_TEETHS[teeth_near_idx]
            heatmap_near_idx = TOOTH_NUM.index(str(teeth_near_idx))
            
            if YES_TEETHS[heatmap_near_idx] == 0:
                continue
            
            heatmap_cur_idx = find_heatmap_channel(teeth_cur_idx, YES_TEETHS)
            heatmap_near_idx = find_heatmap_channel(teeth_near_idx, YES_TEETHS)
            
            cur_heatmap = heatmaps[heatmap_cur_idx]
            near_heatmap = heatmaps[heatmap_near_idx]
            res = cur_heatmap * near_heatmap
            
            losses.append(res)
            
        losses = torch.stack(losses, dim=0)
        losses = losses.mean()
        return losses
=======
        return loss
>>>>>>> parent of de2f1a1... feat : add GD Loss
