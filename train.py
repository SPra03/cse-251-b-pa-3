from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
from torch import optim
import torch.nn.functional as F
from util import *

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():

    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

epochs =20

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = "cpu"
if torch.cuda.is_available():
    device =   "cuda" # TODO determine which device to use (cuda or cpu)
print("Using device: ", device)

# Imp Note!!! Currently Learning rate is kept very high to observe high changes in successive iterations. reduce it in final training
optimizer = optim.Adam(fcn_model.parameters(), lr=0.01)# TODO choose an optimizer
criterion = torch.nn.CrossEntropyLoss()# TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

fcn_model =  fcn_model.to(device)# TODO transfer the model to the device

#################################### Early stopping algorithm
use_early_stopping = True
early_stopping_rounds_const = 5 # aka Patience

def early_stopping(model, iter_num, early_stopping_rounds, best_loss, best_acc, best_iou, best_iter, loss, acc, iou_score, patience):
    if iou_score<=best_iou:
        patience-=1
    else:
        torch.save(fcn_model.state_dict(), "best_model.pth")
        patience = early_stopping_rounds
        best_loss = loss
        best_acc = acc
        best_iou = iou_score
        best_iter = iter_num

    return best_loss, best_acc, best_iou, best_iter, patience

#################################### Early stopping algorithm end

# TODO
def train():
    
    best_iou_score = 0.0

    if use_early_stopping:
        patience = 5
        best_loss = 1e9
        best_acc = -1
        best_iter = 0


    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
            optimizer.zero_grad()


            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device)# TODO transfer the labels to the same device as the model's

            outputs =  fcn_model.forward(inputs) # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss =  criterion(outputs,labels)  #TODO  calculate loss

            loss.backward()
            optimizer.step()

            # TODO  backpropagate

            # TODO  update the weights


            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        val_loss, val_iou, val_acc = val(epoch)
        
        if use_early_stopping:
            best_loss, best_acc, best_iou_score, best_iter, patience = early_stopping(fcn_model, epoch, early_stopping_rounds_const, best_loss, best_acc, best_iou_score,
                                                                best_iter, val_loss, val_acc, val_iou, patience)
            print(f"Patience = {patience}")
            if patience==0:
                print(f"Training stopped early at epoch:{epoch}, best_loss = {best_loss}, best_acc = {best_acc}, best_iou_score = {best_iou_score}, best_iteration={best_iter}")
                break

        # if current_miou_score > best_iou_score:
        #     best_iou_score = current_miou_score
        #     # save the best model
        #     torch.save(fcn_model.state_dict(), "best_model.pth")
    
 #TODO
def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        epoch_loss = 0
        num_iter = 0
        for iter, (inputs, labels) in enumerate(val_loader):
            
            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device)# TODO transfer the labels to the same device as the model's


            outputs = F.log_softmax(fcn_model(inputs), dim=1)
            valoutputs = fcn_model(inputs)
            valloss = criterion(valoutputs, labels)
            epoch_loss += valloss.item()
            num_iter += 1
            _, pred = torch.max(outputs, dim=1)
            mean_iou_scores += [np.mean(iou(pred, labels))]
            accuracy += [pixel_acc(pred, labels)]
            losses += [epoch_loss]

    # print(mean_iou_scores, accuracy)
    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)

 #TODO
def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        epoch_loss = 0
        num_iter = 0
        for iter, (inputs, labels) in enumerate(test_loader):
            
            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device)# TODO transfer the labels to the same device as the model's

            outputs = F.log_softmax(fcn_model(inputs), dim=1)
            valoutputs = fcn_model(inputs)
            valloss = criterion(valoutputs, labels)
            epoch_loss += valloss.item()
            num_iter += 1
            _, pred = torch.max(outputs, dim=1)
            mean_iou_scores += [np.mean(iou(pred, labels))]
            accuracy += [pixel_acc(pred, labels)]
            losses += [epoch_loss]

    # print(mean_iou_scores, accuracy)
    print("Test Performance")
    print(f"Test Loss: is {np.mean(losses)}")
    print(f"Test IoU: is {np.mean(mean_iou_scores)}")
    print(f"Test Pixel acc: is {np.mean(accuracy)}")


    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!



if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()

    print("Loading Best model from best_model.pth as per the IOU score and patience level defined for early stopping..")
    fcn_model.load_state_dict(torch.load("best_model.pth"))

    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()