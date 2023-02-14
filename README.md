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