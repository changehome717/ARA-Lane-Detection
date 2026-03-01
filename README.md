# Angle-Aware Rectangle Anchors for Lane Detection (ARA)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

This repository contains the official PyTorch implementation of *Angle-Aware Rectangle Anchors for Lane Detection: Addressing Geometric Misalignments in Anchor-based Representations*.

## 📝 Introduction
<img width="1728" height="525" alt="10 (1)" src="https://github.com/user-attachments/assets/cb4cfef2-0477-4471-b25b-72e646bc6997" />
Lane detection methods based on line-anchors often suffer from geometric misalignments, specifically Symmetric Point Ambiguity and Magnified Localization Errors, due to continuous point sampling but discrete width and fixed directions. 

To address these issues, we propose **Angle-Aware Rectangle Anchors (ARA)**, a novel representation with continuous width and adaptive directional alignment. Furthermore, we introduce the **Three-Phase Angle-Thresholded Line-Area Transition (TALAT) Loss** to dynamically switch between distance-based and area-based supervision, enabling a smooth transition from coarse to fine geometric alignment.

Our method achieves state-of-the-art (SOTA) or highly competitive performance on TuSimple, CULane, CurveLanes, and LLAMAS benchmarks while maintaining real-time inference speed.


## ⚙️ Installation

### Prerequisites
* Linux
* Python >= 3.8
* PyTorch >= 1.11
* CUDA 11.3

### Environment Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/changehome717/ARA-Lane-Detection
   cd ARA-Lane-Detection
2. Create a virtual environment and install dependencies:
   ```bash
   conda create -n ara python=3.8 -y
   conda activate ara
   pip install -r requirements.txt
   
## 📂 Data Preparation

Our model evaluates on four standard lane detection benchmarks: [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download), [CULane](https://xingangpan.github.io/projects/CULane.html), [CurveLanes](https://github.com/SoulmateB/CurveLanes), and [LLAMAS](https://unsupervised-llamas.com/llamas/).

Please download the datasets from their official websites and organize them as follows:

```text
data/
  ├── TuSimple/
  │   ├── clips/
  │   ├── label_data_*.json
  │   ├── test_tasks_0627.json
  │   ├── test_label.json.json
  ├── CULane/
  │   ├── driver_23_30frame/
  │   ├── driver_161_90frame/
  │   ├── laneseg_label_w16/
  │   ├── list/
  ├── CurveLanes/
  │   ├── train/
  │   ├── valid/
  │   ├── test/
  ├── LLAMAS/
      ├── color_images/train 
      ├── color_images/test 
      ├── color_images/valid 
      ├── labels/train 
      ├── labels/valid

```
For Tusimple, please generate segmentation annotation from the json annotation by:
```bash
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT

```
## 🚀 Model Zoo
We provide pre-trained models for different backbones on the CULane dataset. Download the weights and place them in the weights/ directory.
| Backbone | Dataset | mF1 (%) | F1@50 (%) | FPS | GFLOPs | Download |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-34 | CULane | 56.50 | 81.57 | 112 | 21.5 | [Google Drive](https://drive.google.com/file/d/1HBQGsYcWHfeSZ9ofO7ajSAY0x-FInw5d/view?usp=drive_link) / [Baidu](https://pan.baidu.com/s/1_hcShD_aR4Yw6ARML3Zc1w?pwd=zpgu) |
| ResNet-101| CULane | 57.59 | 81.77 | 46 | 43.0 | [Google Drive](https://drive.google.com/file/d/1cdJTIZYH3kH46ujsxrE9VoSc8WAKJcrb/view?usp=drive_link) / [Baidu](https://pan.baidu.com/s/1NTaet3BICGRlNx9J7aunFw?pwd=he8p) |
| DLA-34 | CULane | 57.41 | 81.92 | 102 | 18.5 | [Google Drive](https://drive.google.com/file/d/1tqjMnhNJT2IWw9QTE7a8JMGq4zSvkT92/view?usp=drive_link) / [Baidu](https://pan.baidu.com/s/1A5SavdEM2IoO76h-k5QcRg?pwd=et2w) |


## 🏃 Getting Started
Training

To train ARA with a specific backbone and dataset, use the provided configuration files. For example, to train the DLA-34 model on CULane:
   ```bash
python main.py configs/ara/ara_culane_dla34.py --gpus 0
```
Evaluation

To evaluate a trained model, specify the path to your downloaded weights:
   ```bash
python main.py configs/ARA/ara_culane_dla34.py --validate --load_from culane_dla34.pth --gpus 0
```
## 🖼️ Visualization

During testing, you can save lane visualization results by enabling `--view`:

```bash
python main.py configs/ARA/ara_culane_dla34.py --validate --view --load_from culane_dla34.pth --gpus 0
```

## 📖 Citation
If you find our work or this code helpful for your research, please consider citing our paper:
   ```bash

@article{,
  title={},
  author={},
  journal={},
  volume={...},
  number={...},
  year={2026}
}
```
## 🤝 Acknowledgements
This project is built upon the excellent work of [CLRNet](https://github.com/Turoad/CLRNet
), [LaneATT](https://github.com/lucastabelini/LaneATT), [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) , [Turoad/lanedet](https://github.com/Turoad/lanedet), [CondLane](https://github.com/aliyun/conditional-lane-detection), [UFLD](https://github.com/cfzd/Ultra-Fast-Lane-Detection), [ZJULearning/resa](https://github.com/ZJULearning/resa), and [CLRerNet](https://github.com/hirotomusiker/CLRerNet). We thank the authors for their open-source contributions.

## 📜 License and Third-Party Notice
This repository includes code derived from third-party open-source projects.  
Please see `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for license texts and attribution details.
