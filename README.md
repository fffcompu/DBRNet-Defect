# DBRNet
DBRNet:Dual-Branch Real-Time Segmentation
NetWork For Metal Defect Detection
![network](./docs/Net.jpg)
paper [[pdf](./docs/samplepaper.pdf)]


## Environment
Python 3.8.10 PyTorch 1.11.0 CUDA 11.3 <br/>
one NVIDIA RTX 3080 GPU
```
conda env create -f requirements.yml
```
## Implementation Details

Using the SGD optimizer with momentum and linear learning rate strategy.
The SGD momentum value was set to 0.9, the initial learning rate was set to
1e-2, the weight decay factor was set to 5e-4. For data augmentation, the NEU-
Seg and MT datasets, we used random augmentation to 0.5 to 2.5 followed
by random cropping to 512×512, and the Severstal Steel Defect Dataset was
randomly cropped to 512×256. The batch size during training was set to 8, all
datasets were divided into train:val:test=6:2:2

## Usage
Download the  three datasets. Put the dataset  to the datacode folder and modify the path in the /datacode/datasetname/<br/>
 <br/>
Train model
```
CUDA_VISIBLE_DEVICES=0 python intertrain.py --model DBRNet  --dataset NEU-Seg --lr 0.01 --epochs 200  --batch_size 8
```
Eval model.
```
python val.py
```
Test model
python test.py
