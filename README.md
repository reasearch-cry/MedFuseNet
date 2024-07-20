# MedFuseNet
Official Pytorch implementation of MedFuseNet: Fusing Local and Global Deep Feature Representations with Hybrid Attention Mechanisms for Medical Image Segmentation

## Architecture:
![model](https://github.com/user-attachments/assets/04619937-c5a6-460d-8ab6-235e6f4bae07)

## Usage:
### Recommended environment:
 ```Python 3.8```

 ```Pytorch 1.11.0```

 ```torchvision 0.12.0```

### Data preparation:
 Synapse Multi-organ dataset: Sign up in the official Synapse website (https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset.

 ACDC dataset: Download the preprocessed ACDC dataset from Google Drive (https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view).

### Training:
For ACDC training run ```python train.py --dataset ACDC```  
For Synapse Multi-organ training run ```python train.py --dataset Synapse```

### Test:
For ACDC test run ```python test.py --dataset ACDC```  
For Synapse Multi-organ test run ```python test.py --dataset Synapse```
