# SemiCVT
The code is for paper: SemiCVT: Semi-Supervised Convolutional Vision Transformer for Semantic Segmentation
## 1.Dataset
Experiments are conducted on two public datasets: Pascal VOC 2012 and Cityscapes.
- **Pascal VOC 2012**  
We evaluate our experiments on both the classic dataset (1, 464 labeled images with 9, 118 unlabeled images), and the blender dataset (10, 582 labeled images), which contains additional 9118 images that be augmented via adopting coarsely annotated 9, 118 images from the SBD dataset).
More details about the dataset split and implementation details will be released till acceptance.  
Refer to this [link](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and download PASCAL VOC 2012 augmented with SBD dataset.

- **Cityscapes**  
The dataset includes 2, 975 and 500 images for training and validation, respectively.  
Download "leftImg8bit_trainvaltest.zip" from: [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)  
Download "gtFine.zip" from: [https://drive.google.com/file/d/10tdElaTscdhojER_Lf7XlytiyAkk7Wlg/view?usp=sharing](https://drive.google.com/file/d/10tdElaTscdhojER_Lf7XlytiyAkk7Wlg/view?usp=sharing)
## 2.Enviorments
- python 3.7
- pytorch 1.9.0
- torchvision 0.10.0

## 3.Train/Test
Train a Fully-Supervised Model  
For instance, we can train a model on PASCAL VOC 2012 with only 1464 labeled data for supervision by:
```
cd experiments/pascal/1464/suponly
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
Or for Cityscapes, a model supervised by only 744 labeled data can be trained by:
```
cd experiments/cityscapes/744/suponly
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
Train a Semi-Supervised Model   
We can train a model on PASCAL VOC 2012 with 1464 labeled data and 9118 unlabeled data for supervision by:
```
cd experiments/pascal/1464/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
Or for Cityscapes, a model supervised by 744 labeled data and 2231 unlabeled data can be trained by:
```
cd experiments/cityscapes/744/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
## 4.Reference
- [U2PL](https://github.com/Haochen-Wang409/U2PL)
