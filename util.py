import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    target[target==255] = 0
    zeros = torch.ones(target.shape)
    target = torch.where(target==255, zeros, target)

    ious = []

    for cls in range(n_classes):
        intersection = torch.sum((pred == cls) & (target == cls))
        union = torch.sum(pred == cls) + torch.sum(target == cls) - intersection
        if union!=0:
            print(cls)
            ious.append(intersection/union)

    ious = np.mean(ious)
    return ious

def pixel_acc(pred, target):
    target[target==255] = 0
    zeros = torch.zeros(target.shape)
    target = torch.where(target==255, zeros, target)
    
    correct = torch.sum(pred==target)
    total_predictions = target.shape[0]*target.shape[1]
    return correct/total_predictions

if __name__=='__main__':
    # to test the utilities
    acc = iou(torch.zeros(5, 5), torch.zeros(5, 5))
    print(f"Pixel Accuracy: {acc}")