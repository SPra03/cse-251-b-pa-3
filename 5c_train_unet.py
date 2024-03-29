import time
from torch.utils.data import DataLoader, ConcatDataset
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import torchvision.transforms as transforms
import util
import numpy as np
from torch import optim
import torch.nn.functional as F
from util import *
from unet_model import *

saveLocation = "./plots/5c_unet/"
if not os.path.exists(saveLocation):
    os.makedirs(saveLocation)

####################### Helper functions

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases


################################ Dataset Loading and processing
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

common_transform = transforms.Compose([
    voc.MirrorFlip(0.5),
    voc.Rotate(10),
    voc.CenterCrop(180)
])

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)
])

target_transform = transforms.Compose([
    MaskToTensor()
])

augmented_train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform, common_transform=common_transform)
augmented_val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform, common_transform=common_transform)
augmented_test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform, common_transform=common_transform)

original_train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
original_val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
original_test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)



train_dataset = ConcatDataset([augmented_train_dataset, original_train_dataset])
val_dataset = ConcatDataset([augmented_val_dataset, original_val_dataset])
test_dataset = ConcatDataset([augmented_test_dataset, original_test_dataset])

# Parameters to optimize the dataset loading, which was the slowest step
NUM_WORKERS = 4
PREFETCH_FACTOR = 2 # improves data transfer speed between GPU and CPU and reduces GPU wait time
train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)

#################################### End of Dataset Loading and processing

epochs = 100

n_class = 21

fcn_model = UNET(n_class=n_class)
fcn_model.apply(init_weights)

device = "cpu"
if torch.cuda.is_available():
    device =   "cuda" #  determine which device to use (cuda or cpu)
print("Using device: ", device)

# Imp Note!!! Currently Learning rate is kept very high to observe high changes in successive iterations. reduce it in final training
optimizer = optim.Adam(fcn_model.parameters(), lr=0.005)#  choose an optimizer

#################################### Loss criterion

def getClassWeights(train_dataset):
        # Define a dictionary to store the pixel counts for each class
    class_pixel_counts = {i: 0 for i in range(n_class)}

    # Iterate over the train dataset and count the number of pixels for each class in the mask
    for i in range(len(train_dataset)):
        img, mask = train_dataset[i]
        for c in range(n_class):
            class_pixel_counts[c] += torch.sum(mask == c)

    # Print the frequency for each class
    # for c in range(n_class):
    #     frequency = class_pixel_counts[c] / torch.sum(torch.Tensor(list(class_pixel_counts.values())))
        # print(f"Class {c}: {frequency}")

    imbalance_freq = np.array(list(class_pixel_counts.values()))
    # print(imbalance_freq)
    imbalance_weights = 1-imbalance_freq/np.sum(imbalance_freq)
    # imbalance_weights = imbalance_weights/np.sum(imbalance_weights)
    print("Imbalance weights are: ", imbalance_weights)

    return imbalance_weights

use_weights = True

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=self.weight)
        return loss

if use_weights:
    imbalance_weights = getClassWeights(train_dataset)
    imbalance_weights = torch.Tensor(imbalance_weights).to(device)
    criterion = WeightedCrossEntropyLoss(weight=imbalance_weights)
else:
    criterion = torch.nn.CrossEntropyLoss()#  Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

#####################

fcn_model =  fcn_model.to(device)#  transfer the model to the device

#################################### Early stopping algorithm
use_early_stopping = True
early_stopping_rounds_const = 5 # aka Patience

def early_stopping(model, iter_num, early_stopping_rounds, best_loss, best_acc, best_iou, best_iter, loss, acc, iou_score, patience):
    if iou_score<=best_iou:
        patience-=1
    else:
        torch.save(fcn_model.state_dict(), saveLocation+"best_model.pth")
        patience = early_stopping_rounds
        best_loss = loss
        best_acc = acc
        best_iou = iou_score
        best_iter = iter_num

    return best_loss, best_acc, best_iou, best_iter, patience

#################################### Early stopping algorithm end


def train():

    torch.autograd.set_detect_anomaly(True)
    
    best_iou_score = 0.0

    if use_early_stopping:
        patience = 5
        best_loss = 1e9
        best_acc = -1
        best_iter = 0

    trainEpochLoss = []
    trainEpochAccuracy = []
    trainEpochIOU = []
    valEpochLoss = []
    valEpochAccuracy = []
    valEpochIOU = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0, last_epoch=-1)

    for epoch in range(epochs):

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        train_loss = []
        train_acc = []
        train_iou = []

        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            #   reset optimizer gradients
            optimizer.zero_grad()


            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)#  transfer the input to the same device as the model's
            labels =   labels.to(device)#  transfer the labels to the same device as the model's

            trainOutputs =  fcn_model.forward(inputs) #   Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!
            trainOutputs = F.softmax(trainOutputs)
            loss =  criterion(trainOutputs,labels)  #  calculate loss
            loss.backward()

            with torch.no_grad():
                # To compute accuracy and IOU
                # outputs = F.log_softmax(fcn_model(inputs), dim=1)
                _, pred = torch.max(trainOutputs, dim=1)
                
                train_iou += [np.mean(iou(pred, labels))]
                train_acc += [pixel_acc(pred, labels)]
                train_loss.append(loss.item())

            optimizer.step()

            if iter % 10 == 0:
                print(f"==> epoch{epoch}, iter{iter}, Train set=> loss: {train_loss[-1]}, IOU: {train_iou[-1]}, Acc: {train_acc[-1]}")

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        val_loss, val_iou, val_acc = val(epoch)
        
        if use_early_stopping:
            best_loss, best_acc, best_iou_score, best_iter, patience = early_stopping(fcn_model, epoch, early_stopping_rounds_const, best_loss, best_acc, best_iou_score,
                                                                best_iter, val_loss, val_acc, val_iou, patience)
            print(f"Patience = {patience}")
            if patience==0:
                print(f"==> Training stopped early at epoch:{epoch}, best_loss = {best_loss}, best_acc = {best_acc}, best_iou_score = {best_iou_score}, best_iteration={best_iter}")
                break
        scheduler.step()
        print(f"Scheduler Learning Rate: {scheduler.get_last_lr()}")
        
        ##### Plotting values
        trainEpochLoss.append(np.mean(np.asarray(train_loss)))
        trainEpochIOU.append(np.mean(np.asarray(train_iou)))
        trainEpochAccuracy.append(np.mean(train_acc))
        valEpochLoss.append(val_loss)
        valEpochIOU.append(val_iou)
        valEpochAccuracy.append(val_acc)

    plots(trainEpochLoss, trainEpochAccuracy, trainEpochIOU, valEpochLoss, valEpochAccuracy, valEpochIOU, best_iter, saveLocation=saveLocation)

    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        num_iter = 0
        for iter, (inputs, labels) in enumerate(val_loader):
            
            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)#  transfer the input to the same device as the model's
            labels =   labels.to(device)#  transfer the labels to the same device as the model's


            outputs = F.softmax(fcn_model(inputs), dim=1)
#             valoutputs = fcn_model(inputs)
            valloss = criterion(outputs, labels)
            
            num_iter += 1
            _, pred = torch.max(outputs, dim=1)
            mean_iou_scores += [np.mean(iou(pred, labels))]
            accuracy += [pixel_acc(pred, labels)]
            losses += [valloss.item()]

    # print(mean_iou_scores, accuracy)
    print(f"=========> Loss at epoch {epoch} is {np.mean(losses)}")
    print(f"=========> IoU at epoch {epoch} is {np.mean(mean_iou_scores)}")
    print(f"=========> Pixel acc at epoch {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)


def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        num_iter = 0
        for iter, (inputs, labels) in enumerate(test_loader):
            
            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)#  transfer the input to the same device as the model's
            labels =   labels.to(device)#  transfer the labels to the same device as the model's

            outputs = F.softmax(fcn_model(inputs), dim=1)
#             valoutputs = fcn_model(inputs)
            valloss = criterion(outputs, labels)
            num_iter += 1
            _, pred = torch.max(outputs, dim=1)
            mean_iou_scores += [np.mean(iou(pred, labels))]
            accuracy += [pixel_acc(pred, labels)]
            losses += [valloss.item()]

            if iter<=5:
                index = 3
                plot_image_segMaps(inputs, pred, labels, iter, index, saveLocation=saveLocation)

    # print(mean_iou_scores, accuracy)
    print("Test Performance")
    print(f"Test Loss: is {np.mean(losses)}")
    print(f"Test IoU: is {np.mean(mean_iou_scores)}")
    print(f"Test Pixel acc: is {np.mean(accuracy)}")


    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!



if __name__ == "__main__":

    if use_weights:
        print("Imbalance weights are: ", imbalance_weights)
    val(0)  # show the accuracy before training
    train()

    print(f"Loading Best model from {saveLocation}best_model.pth as per the IOU score and patience level defined for early stopping..")
    fcn_model.load_state_dict(torch.load(saveLocation+"best_model.pth"))

    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
