import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    target[target==255] = 0

    ious = []

    for cls in range(n_classes):
        intersection = torch.sum((pred == cls) & (target == cls)).item()
        union = torch.sum(pred == cls) + torch.sum(target == cls) - intersection
        union = union.item()
        if union!=0:
            ious.append(intersection/union)

    ious = np.array(ious)
    return ious

def pixel_acc(pred, target):
    target[target==255] = 0
    
    correct = torch.sum(pred==target).item()
    total_predictions = target.shape[0]*target.shape[1]*target.shape[2]
    return correct/total_predictions

if __name__=='__main__':
    # to test the utilities
    acc = iou(torch.zeros(16, 224, 224), torch.zeros(16, 224, 224))
    print(f"Pixel Accuracy: {acc}")