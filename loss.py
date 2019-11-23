import torch
import torch.nn.functional as F

# Loss functions
def loss_final(pred, label, loss_type, criteria, save_rate=0.9, gamma=2.0):
    if loss_type == 'Softmax':
        loss_final = criteria(pred, label)
    elif loss_type == 'FocalLoss':
        assert (gamma >= 0)
        input = F.cross_entropy(pred, label, reduce=False)
        pt = torch.exp(-input)
        loss = (1 - pt) ** gamma * input
        loss_final = loss.mean()
    elif loss_type == 'HardMining':
        batch_size = pred.shape[0]
        loss = F.cross_entropy(pred, label, reduce=False)
        ind_sorted = torch.argsort(-loss) # from big to small
        num_saved = int(save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(F.cross_entropy(pred[ind_update], label[ind_update]))
    else:
        raise Exception('unknown loss type!!')

    return loss_final




