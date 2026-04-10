# Frequency-enhanced Contextual Conversion Network for Esophageal Lesion Segmentation

## 1. Preface

- This repository provides code for "_**Frequency-enhanced Contextual Conversion Network for Esophageal Lesion Segmentation**_" 


## 2. Overview

### 2.1. Introduction

Esophageal cancer (EC) is a prevalent disease with a high fatality rate, making early detection in clinical practice challenging. Similar to identifying colon polyps, doctors often utilize an endoscope to observe EC lesions. However, segmenting early esophageal cancer is difficult, due to the flat shape, similar color to the surrounding mucosa, and the interference of esophageal inflammation. Moreover, this field lacks a comprehensive benchmark for depth exploration. To address these challenges, we construct a new benchmark and propose a novel Frequency-enhanced Contextual Conversion Network (FCC-Net) for esophageal lesion segmentation, which effectively captures multi-level context information and leverages the frequency domain to identify fine-grained lesion boundaries. Specifically, we propose a Frequency-enhanced Context Module (FCM) to merge high-level features, facilitating the capture of global context information and offering essential cues from the frequency domain for accurate lesion boundary identification. Additionally, we present a Scale-induced Cross-level Fusion (SCF) module that fully integrates features from the adjacent levels. This enables the learning of multi-scale features through different convolution kernels, effectively handling scale variations. Furthermore, a Contextual Conversion Module (CCM) is presented to incorporate the global context information and boundary-related cues to enhance EC lesion segmentation performance. Experimental results across multiple EC datasets demonstrate that our model outperforms other state-of-the-art medical segmentation methods. 

### 2.2. Dataset Overview

<p align="center">
    <img src="dataset_example.png"/> <br />
    <em>
    Figure 1: Illustrations of some examples from NBI and WLI datasets with a diversity of challenging factors, such as varying sizes, irregular shapes, and blurred edges (red or green circled area), and some benign changes similar to the appearance of esophageal cancer (blue circled areas).
    </em>
</p>

## 3.Training and Inference
Train
```
python MyTrain_val.py 
```

Test
```
python MyTesting.py 
```

## 4. Citation
```
@article{tang2025frequency,
  title={Frequency-enhanced contextual conversion network for esophageal lesion segmentation},
  author={Tang, Ziqi and Niu, Xiaotong and Rong, Long and Zhang, Yizhe and Bi, Yawei and Ru, Nan and Li, Longsong and Chai, Ningli and Zhou, Tao},
  journal={Pattern Recognition},
  pages={112235},
  year={2025},
  publisher={Elsevier}
}
```