# Yeast Cell Segmentation with Generalist Deep Neural Network Mediar-Former 

**Group members:** Yanruiqi Yang(#346510), Yasmine Chaker(#), Ameer Elkhayat(#)

## Data preparation:

1. Training Dataset:
   906 yeast cell microscopy images with labels, provided by Prof. Sahand Jamal Rahi (Laboratory of the Physics of Biological Systems, LPBS), in the format of .tif;
2. Evaluation Dataset:
   The [Yeast Image Toolkit](http://yeast-image-toolkit.biosim.eu/pmwiki.php) (Dataset 1, Dataset 3)

   
## Model: 

- Pretrained model: [Mediar-Former](https://github.com/Lee-Gihun/MEDIAR) is the "1st winner" in the [NeurIPS-2022 Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/)
- Predictor: Ensembled with the model and fine-tuned generalist cell segmentation model

## Environment Setup:

## Training:

## Evaluation:

## Conclusion:

## Reference:

1. [MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy](https://arxiv.org/abs/2212.03465)
2. [YeastNet: Deep-Learning-Enabled Accurate Segmentation of Budding Yeast Cells in Bright-Field Microscopy](https://www.mdpi.com/2076-3417/11/6/2692/htm)
