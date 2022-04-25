# Trajectory Prediction with Graph-based Dual-scale Context Fusion

## Introduction
This is the project page of the paper 
```
Lu Zhang, Peiliang Li, Jing Chen and Shaojie Shen, "Trajectory Prediction with Graph-based Dual-scale Context Fusion", 2021.
```
* Notice: The code will be released after the publishing of this paper.

**Preprint:** [Link](https://arxiv.org/abs/2111.01592)

<p align="center">
  <img src="files/cover.png" width = "500"/>
</p>

## Have a try!

### Install dependencies
- Create a new conda env
```
conda create --name dsp python=3.8
conda activate dsp
```

- Install PyTorch according to your CUDA version. For RTX 30 series, we recommend CUDA >= 11.1, PyTorch >= 1.8.0.
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

- Install Argoverse API, please follow this [page](https://github.com/argoai/argoverse-api).

- Install other dependencies
```
pip install scikit-image IPython tqdm ipdb tensorboard
```

- Install PyTorch Scatter, please refer to this [page](https://github.com/rusty1s/pytorch_scatter).


### Play with pretrained models
Download the pretrained model and preprocessed data samples.

### Train from scratch


**Quantitative Results:**

<p align="center">
  <img src="files/quant_res.png" width = "800"/>
</p>

**Video:**
<a href="https://youtu.be/AifLEhVQXjo" target="_blank">
  <p align="center">
    <img src="files/video_cover.png" alt="video" width="640" height="360" border="10" />
  </p>
</a>

**Supplementary Video (Argoverse Tracking dataset):**
<a href="https://youtu.be/Rjk2u9O59R4" target="_blank">
  <p align="center">
    <img src="files/vid2_cover.png" alt="video" width="640" height="360" border="10" />
  </p>
</a>

## Demo
* Color scheme: green - predicted trajectories; red - observation & GT trajectories; orange - other agents.

<p align="center">
  <img src="files/1.gif" width = "400" height = "400"/>
  <img src="files/2.gif" width = "400" height = "400"/>
  <img src="files/3.gif" width = "400" height = "400"/>
  <img src="files/4.gif" width = "400" height = "400"/>
</p>

## Citation
If you find this paper useful for your research, please consider citing the following:
```
@article{zhang2021trajectory,
  title={Trajectory Prediction with Graph-based Dual-scale Context Fusion},
  author={Zhang, Lu and Li, Peiliang and Chen, Jing and Shen, Shaojie},
  journal={arXiv preprint arXiv:2111.01592},
  year={2021}
}
```