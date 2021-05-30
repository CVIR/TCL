## Project Page of Semi-Supervised Action Recognition with Temporal Contrastive Learning [[Paper]](https://arxiv.org/pdf/2102.02751.pdf)

This repository contains the implementation details of our Temporal Contrastive Learning (TCL) approach for action recognition in videos.

Ankit Singh*, Omprakash Chakraborty*, Ashutosh Varshney, Rameswar Panda, Rogerio Feris, Kate Saenko and Abir Das, "Semi-Supervised Action Recognition with Temporal Contrastive Learning"
*: Equal contributions

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{singh2021semi,
    title={Semi-Supervised Action Recognition with Temporal Contrastive Learning},
    author={Singh, Ankit and Chakraborty, Omprakash and Varshney, Ashutosh and Panda, Rameswar and Feris, Rogerio and Saenko, Kate and Das, Abir},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021},
    month = jun
}
```

### Prerequisites
- Python 3.X
- PyTorch 1.X
- torchvision
- pyyaml

For video data pre-processing, you may need ffmpeg.


## Requirements
```
pip install -r requirements.txt
```

## Python script overview

`main.py` - It contains the code for Temporal Contrastive Learning(TCL) with the 2 pathway model.

`opts.py` - It contains the file with default value for different parameter used in 2 pathway model.

`ops/dataset_config.py` - It contains the code for different config for different dataset and their location e.g Kinetics, Jester, SomethingV2

`ops/dataset.py` - It contains the code for how frames are sampled from video

### Key Parameters:
 `use_group_contrastive`: to use group contrastive loss \
 `use_finetuning` : option to use finetuning at the last \
 `finetune_start_epoch`: from which epoch to start finetuning \
 `finetune_lr`: if want to use different lr other than normal one\
 `gamma_finetune`: weight for pl_loss in finetuning step \
 `finetune_stage_eval_freq`: printing freq for finetuning stage\
 `threshold`: used in fine tuning step for selection of labels \
 `sup_thresh`: till which epoch supervised only to be run \
 `percentage`: percentage of unlabeled data e.g 0.99 ,0.95 \
 `gamma`: weight of instance contrastive loss \
 `lr`: starting learning rate \
 `mu`: ratio of unlabeled to labeled data \
 `flip`: whether to use horizontal flip in transforms or not


### Creating Dataset
The `root_dataset.yaml` must contains the root folder of the particular dataset.
Each dataset folder must contains two sub-folders.
- Frames: This sub-folder contains the extracted Frames of the videos from the dataset. To extract frames from videos, Please refer to [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation)
- data: This sub-folder contains the training,validation split files along with the file containing the list of all the classes of the videos.These names should be consistent with `ops/dataset_config.py`. Samples files are provided in the dataset folder.

### Training TCL
- For running `x%` labeled data scenario, it expects to have a folder named `Run_x` where all the labeled and unlabeled data will be split as per the input seed.
- All the models and logs will be stored inside a sub folder of checkpoints directory. A different subfolder will be created on each execution.

### Sample Code to Run for 2 stream final approach:

`python main.py somethingv2 RGB --seed 123 --strategy classwise
 --arch resnet18 --num_segments 8 --second_segments 4 --threshold 0.8 --gd 20 --lr 0.02 --wd 1e-4 
 --epochs 400 --percentage 0.95 --batch-size 8 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --print-freq 50
 --shift --shift_div=8 --shift_place=blockres --npb --gpus 0 1  --mu 3 --gamma 9 --gamma_finetune 1 
--use_group_contrastive --use_finetuning --finetune_start_epoch 350 --sup_thresh 50 --valbatchsize 16 --finetune_lr 0.002`

## Reference

1. Lin, Ji, Chuang Gan, and Song Han. "Tsm: Temporal shift module for efficient video understanding." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
