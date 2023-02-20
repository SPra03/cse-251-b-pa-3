# cse-251-b-pa-3

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

#### Baseline with improvements 3, 4

Files - train.py and basic_fcn.py

```bash
python train.py
```

#### Our Custom model (5a)

Files - model_experiment_5a.py and train_experiment_5a.py

```bash
python train_experiment_5a.py
```

#### Transfer learning (5b)

Files - transfer.py and transfer_model.py

```bash
python transfer.py
```

#### U-Net (5c)

Files - unet.py and unet_model.py

```bash
python train.py
```
