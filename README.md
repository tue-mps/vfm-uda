# Exploring the Benefits of Vision Foundation Models for Unsupervised Domain Adaptation (CVPR 2024 Second Workshop on Foundation Models)
**Authors:** Bruno B. Englert, Fabrizio J. Piva, Tommie Kerssies, Daan de Geus, Gijs Dubbelman  
**Affiliation:** Eindhoven University of Technology  
**Publication:** CVPR 2024 Workshop Proceedings for the Second Workshop on Foundation Models  
**Paper:** [arXiv](http://arxiv.org/abs/2406.09896)  
**Code**: [GitHub](https://github.com/tue-mps/vfm-uda)

🔔 **News:**
* [2025-03-30] We are happy to announce that our analysis work ["What is the Added Value of UDA in the VFM Era?"](https://arxiv.org/abs/2504.18190) was accepted at **CVPRW25**.
* [2025-03-11] We are happy to announce that our work [VFM-UDA++](https://arxiv.org/abs/2503.10685), an improvement of VFM-UDA, was released on arXiv.


## Abstract
Achieving robust generalization across diverse data domains remains a significant challenge in computer vision. This challenge is important in safety-critical applications, where deep-neural-network-based systems must perform reliably under various environmental conditions not seen during training. Our study investigates whether the generalization capabilities of Vision Foundation Models (VFMs) and Unsupervised Domain Adaptation (UDA) methods for the semantic segmentation task are complementary. Results show that combining VFMs with UDA has two main benefits: (a) it allows for better UDA performance while maintaining the out-of-distribution performance of VFMs, and (b) it makes certain time-consuming UDA components redundant, thus enabling significant inference speedups. Specifically, with equivalent model sizes, the resulting VFM-UDA method achieves an 8.4x speed increase over the prior non-VFM state of the art, while also improving performance by +1.2 mIoU in the UDA setting and by +6.1 mIoU in terms of out-of-distribution generalization. Moreover, when we use a VFM with 3.6x more parameters, the VFM-UDA approach maintains a 3.3x speed up, while improving the UDA performance by +3.1 mIoU and the out-of-distribution performance by +10.3 mIoU. These results underscore the significant benefits of combining VFMs with UDA, setting new standards and baselines for Unsupervised Domain Adaptation in semantic segmentation.

## Installation 
1. **Create a Weights & Biases (W&B) account.**
   - The metrics during training are visualized with W&B: https://wandb.ai 


2. **Environment setup.**
     ```bash 
    conda create -n fuda python=3.10 && conda activate fuda
    ```

3. **Install required packages.**
    ```bash
    pip install -r requirements.txt
    ```

## Data preparation
 - **Cityscapes**: [Download 1](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Download 2](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
 - **GTA V**: [Download 1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip) | [Download 2](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_images.zip) | [Download 3](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_images.zip) | [Download 4](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_images.zip) | [Download 5](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_images.zip) | [Download 6](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_images.zip) | [Download 7](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_images.zip) | [Download 8](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_images.zip) | [Download 9](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip) | [Download 10](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip) | [Download 11](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) | [Download 12](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip) | [Download 13](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip) | [Download 14](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip) | [Download 15](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip) | [Download 16](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip) | [Download 17](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip) | [Download 18](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip) | [Download 19](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip) | [Download 20](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip)
 - **Mapillary**: [Download 1](https://www.mapillary.com/dataset/vistas)
 - **WildDash**: [Download 1](https://wilddash.cc/download/wd_public_02.zip) (Download the "old WD2 beta", not the new "Public GT Package")
   - For WilDdash, an extra step is needed to create the train/val split. After the "wd_public_02.zip" is downloaded, place the files from the "wilddash_trainval_split" in the same direcetory as the zip file. After that, run:
     ```bash 
     chmod +x create_wilddash_ds.sh
     ./create_wilddash_ds.sh
     ```
     This creates a new zip files, which should be used during training.
        
All the zipped data should be placed under one directory. No unzipping is required.



## Usage
### Training
We recommend using 4 GPUs with 2 batch size per GPU. On a A100, training a ViT-L will take around 20h.

   ```bash
   python main.py fit -c uda_vit_vanilla.yaml --root /data  --trainer.devices [0,1,2,3]
   ```
   (replace ```/data``` with the folder where you stored the datasets)

  We note that there are small variations in performance between training runs, due to the stochasticity in the process, particularly for UDA techniques. Therefore, results may differ slightly depending on the random seed.’


### Evaluating
To evaluate a pre-trained VFM-UDA++ model, run:

```bash
python3 main.py validate -c uda_vit_vanilla.yaml --root /data  --trainer.devices "[0]" --model.network.ckpt_path "/path/to/checkpoint.ckpt"
```
or use huggingface urls directly
```bash
python3 main.py validate -c uda_vit_vanilla.yaml --root /data  --trainer.devices "[0]" --model.network.ckpt_path "https://huggingface.co/tue-mps/vfmuda_base_gta2city/resolve/main/vfmuda_base_gta2city_771miou_step40000.ckpt"
```

(replace ```/data``` with the folder where you stored the datasets)



## Model Zoo
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">Cityscapes (miou)</th>
<th valign="bottom">WildDash2 (miou)</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->

<tr>
<td align="left">VFM-UDA</td>
<td align="center">ViT-B</td>
<td align="center">DINOv2</td>
<td align="center">77.1</td>
<td align="center">60.8</td>
<td align="center"><a href="https://huggingface.co/tue-mps/vfmuda_base_gta2city/resolve/main/vfmuda_base_gta2city_771miou_step40000.ckpt">model</a></td>
</tr>

<tr>
<td align="left">VFM-UDA</td>
<td align="center">ViT-L</td>
<td align="center">DINOv2</td>
<td align="center">78.4</td>
<td align="center">64.7</td>
<td align="center"><a href="https://huggingface.co/tue-mps/vfmuda_large_gta2city/resolve/main/vfmuda_large_gta2city_784miou_step40000.ckpt">model</a></td>
</tr>

</tbody></table>

*Note: these models are re-trained, so the results differ slightly from those reported in the paper.*

## Citation
```
@inproceedings{englert2024exploring,
  author={{Englert, Brunó B.} and {Piva, Fabrizio J.} and {Kerssies, Tommie} and {de Geus, Daan} and {Dubbelman, Gijs}},
  title={Exploring the Benefits of Vision Foundation Models for Unsupervised Domain Adaptation},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024},
}
```

## Acknowledgement
We use some code from:
 * DINOv2 (https://github.com/facebookresearch/dinov2): Apache-2.0 License
 * Masked Image Consistency for Context-Enhanced Domain Adaptation (https://github.com/lhoyer/MIC): Copyright (c) 2022 ETH Zurich, Lukas Hoyer, Apache-2.0 License
 * SegFormer (https://github.com/NVlabs/SegFormer): Copyright (c) 2021, NVIDIA Corporation, NVIDIA Source Code License
 * DACS (https://github.com/vikolss/DACS): Copyright (c) 2020, vikolss, MIT License 
 * MMCV (https://github.com/open-mmlab/mmcv): Copyright (c) OpenMMLab, Apache-2.0 License
