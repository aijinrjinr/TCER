# Self-supervised deep learning of gene-gene interactions for improved gene expression recovery (TCER)


# Paper
## Pipeline
<img src="imgs/pipeline.png" style="zoom:38%;" />

## ER-Net Structure
<img src="imgs/network_structure.png" style="zoom:38%;" />

# Code
## Requirement

## Data
Please refer to the following Google Drive link to download the cellular taxonomy dataset with 0.5% efficiency loss. Notice that the provided data has already been constructed into GenoMap from the raw data.

https://drive.google.com/file/d/1H1tpwM96IR21qTKYF3EK5m6ziAVwUKIp/view?usp=share_link

### How to use the data?
```
data = loadmat('CellularTax_dataSAVER10-2000.mat')
train_genomaps = data["train_genoMaps"]   #  ==> w x h x c x #cells
train_genomaps_GT = data["train_genoMaps_GT"] # reference data for genomaps in the training set
test_genomaps = data["test_genomaps"] 
test_genomaps_GT = data["test_genomaps_GT"]  # reference data for genomaps in the test set
```
## Train
To train our ER-Net from scratch, please use the following command.
```
python train_genoMap.py --dataset 'CellularTax' --rate '10-2000' --epochs 50
```
## Test
To test the trained ER-Net, please use the following command.
```
python test_genoMap.py --dataset 'CellularTax' --rate '10-2000' --epoch_suffix 50 --model_path XXXX
```
You can alo download our trained model of the cellular taxonomy dataset with 0.5% efficiency loss for testing in the following Google Drive link.

https://drive.google.com/file/d/1xNwxv4MrJLEldvgFaqf7cJHakxmaPyDr/view?usp=share_link

## Visualization
The visualization tutorial could be found in the ```Visulization.ipynb``` file. 

Our imputation results and the corresponding cell types could be found in the following Google Drive link.

https://drive.google.com/drive/folders/1t84QTYd8DvfpE2z2XT1SzXT0LnrBY7p9?usp=share_link

