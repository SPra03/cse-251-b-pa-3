# cse-251-b-pa-3

## Link to documentation: 
https://drive.google.com/file/d/14NzQen3zb1pAOOlqBiOiY-6EiQPH2N-M/view?usp=share_link

## Note
The results will all be visible in the plots directory upon running. Each result plot, model, etc will be saved inside a different sub-directory. The names clearly indicate the names of the experiment.


### Setup
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

### Dataset Download

```bash
python download.py
cd data
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```


## Experiment vs Files

#### Baseline
Model file: basic_fcn.py
```bash
python 3_train.py
```
#### Baseline with improvements 4a - Learning rate scheduling
Model file: basic_fcn.py
```bash
python 4a_train.py
```

#### Baseline with improvements 4b - Data Augmemtation
Model file: basic_fcn.py
```bash
python 4b_train.py
```

#### Baseline with improvements 4c - Imbalance class weights
Model file: basic_fcn.py
```bash
python 4c_train.py
```

#### Our Custom model (5a) - Advanced FCN

Model - model_experiment_5a.py

```bash
python 5a_train_experiment.py
```

#### Transfer learning (5b)

Model - transfer_model.py

```bash
python 5b_train_transfer.py
```

#### U-Net (5c)

Model - unet_model.py

```bash
python 5c_train_unet.pu
```

### Description of miscellaneous files:
voc.py - Dataset class for loading VOC dataset
utils.py - Metrics for IoU and pixel accuracy

### Zipping plots directory

```bash
tar -czvf DL_PA3.tar.gz --exclude='*.pth' plots/
```
