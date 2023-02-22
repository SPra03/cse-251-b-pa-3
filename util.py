import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Average over classes
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

def plots(trainEpochLoss, trainEpochAccuracy, trainEpochIOU, valEpochLoss, valEpochAccuracy, valEpochIOU, earlyStop, saveLocation="./plots/"):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """
    
    # saveLocation = "./plots/"+saveLocation
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
        print("Created Plots directory.")
    
    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(saveLocation+"loss.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(saveLocation+"accuracy.png")
    plt.close(fig2)

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochIOU, 'r', label="Training IOU")
    ax2.plot(epochs, valEpochIOU, 'g', label="Validation IOU")
    plt.scatter(epochs[earlyStop], valEpochIOU[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('IOU Score Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('IOU Score', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(saveLocation+"iou.png")
    plt.close(fig2)

def plot_segmentation_map(segmentation_map, fname):
    segmentation_map = segmentation_map.to("cpu")
    # Create a color map from the given palette
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
    #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
    #class 1 and so on......
    plt.clf()
    cmap = plt.cm.colors.ListedColormap(np.array(palette).reshape(-1, 3)/255)
    
    # Plot the segmentation map using the color map
    plt.imshow(segmentation_map, cmap=cmap)
    plt.axis('off')
    plt.savefig(fname)

def plot_image_segMaps(inputs, pred, labels, iter, index, saveLocation):
    plt.clf()
    # print(inputs.shape, pred.shape, labels.shape)
    #print(pred[index, :, :].tolist(), "\n", labels[index, :, :].tolist())
    plt.imshow(inputs[index, :, :, :].transpose(0, 2).transpose(0, 1).to("cpu"))
    plt.savefig(saveLocation+f"{iter}_image.png")
    plot_segmentation_map(pred[index, :, :], saveLocation+f"{iter}_prediction.png")
    plot_segmentation_map(labels[index, :, :], saveLocation+f"{iter}_label.png")

if __name__=='__main__':
    # to test the utilities
    tensor = torch.zeros(16, 224, 224)
    acc = iou(tensor, tensor)
    print(f"Pixel Accuracy: {acc}")
    plot_segmentation_map(tensor[0, :, :])
