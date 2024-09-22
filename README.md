# [TTT-Unet](https://your-link-to-paper-or-project)

Official repository for **TTT-Unet: Enhancing U-Net with Test-Time Training Layers for Biomedical Image Segmentation**. This code is based on **U-Mamba** and serves as the official implementation of our paper.

## Installation 

The setup for TTT-Unet follows the same installation and configuration as **U-Mamba**. Please refer to the [U-Mamba repository](https://github.com/bowang-lab/U-Mamba) for detailed setup instructions.

1. Clone this repository: 
   ```bash
   git clone https://github.com/rongzhou7/TTT-Unet.git
2. Follow the installation steps provided in [U-Mamba](https://github.com/bowang-lab/U-Mamba).

## Model Training

The model training process for TTT-Unet also follows the same procedures as U-Mamba. For data preparation and model training, please refer to the [U-Mamba repository](https://github.com/bowang-lab/U-Mamba).

## Inference

Inference for TTT-Unet models also follows U-Mamba’s setup. To generate predictions, use the `nnUNetv2_predict` command with the appropriate configuration. For further details, refer to the [U-Mamba repository](https://github.com/bowang-lab/U-Mamba).

## Paper

If you use our work in your research, please cite our paper as follows:

```bibtex
@article{zhou2024ttt,
  title={TTT-Unet: Enhancing U-Net with Test-Time Training Layers for biomedical image segmentation},
  author={Zhou, Rong and Yuan, Zhengqing and Yan, Zhiling and Sun, Weixiang and Zhang, Kai and Li, Yiwei and Ye, Yanfang and Li, Xiang and He, Lifang and Sun, Lichao},
  journal={arXiv preprint arXiv:2409.11299},
  year={2024}
}

## Acknowledgements

This project is based on **U-Mamba**. We acknowledge all the authors of the employed public datasets, as well as the authors of [U-Mamba](https://github.com/bowang-lab/U-Mamba) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for making their valuable code publicly available.
