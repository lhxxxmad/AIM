# AIM: Adapting Image Models for Efficient Video Understanding

This repo is the official implementation of ["AIM: Adapting Image Models for Efficient Video Understanding"](https://openreview.net/forum?id=CIoSZ_HKHS7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)) at ICLR 2023.

If you find our work useful in your research, please cite:
```
@inproceedings{
    yang2023aim,
    title={AIM: Adapting Image Models for Efficient Video Understanding},
    author={Taojiannan Yang and Yi Zhu and Yusheng Xie and Aston Zhang and Chen Chen and Mu Li},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=CIoSZ_HKHS7}
}
```

## Introduction

In this work, we propose a novel method to Adapt pre-trained Image Models (AIM) for efficient video understanding. By freezing the pre-trained image model and adding a few lightweight Adapters, we introduce spatial adaptation, temporal adaptation and joint adaptation to gradually equip an image model with spatiotemporal reasoning capability. The overall structure of the proposed method is shown in the figure below.

<p><img src="figures/overallstructure.png" width="800" /></p>

During training, only Adapters are updated, which largely saves the training cost while still achieve competitive performance with SoTA full finetuned video models. As shown in the figure below, AIM outperforms previous SoTA methods while using less number of tunable parameters and inference GFLOPs.

<p><img src="figures/overallperformance.png" width="500" /></p>

## Installation

The codes are based on [VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer), which is based on [MMAction2](https://github.com/open-mmlab/mmaction2). To prepare the environment, please follow the following instructions.
```shell
# create virtual environment
conda create -n AIM python=3.7.13
conda activate AIM

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install other requirements
pip install -r requirements.txt

# install mmaction2
python setup.py develop
```

## Data Preparation
The codes are based on [MMAction2](https://github.com/open-mmlab/mmaction2). You can refer to [MMAction2](https://github.com/open-mmlab/mmaction2) for a general guideline on how to prepare the data. All the datasets (K400, K700, SSv2 and Diving-48) used in this work are supported in [MMAction2](https://github.com/open-mmlab/mmaction2).

## Training
The training configs of different experiments are provided in `configs/recognition/vit/`. To run experiments, please use the following command. `PATH/TO/CONFIG` is the training config you want to use.
```shell
bash tools/dist_train.sh PATH/TO/CONFIG 4 --test-last --validate --cfg-options work_dir=PATH/TO/OUTPUT
```
We also provide a training script in `run_exp.sh`. You can simply change the training config to train different models.

### Apex (optional):
We use apex for mixed precision training by default. To install apex, please follow the instructions in the [repo](https://github.com/NVIDIA/apex).

If you would like to disable apex, comment out the following code block in the [configuration files](configs/recognition/vit/):
```
# do not use mmcv version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

