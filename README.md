## Semi-Supervised Action Recognition with Temporal Contrastive Learning [[Paper]](https://arxiv.org/pdf/2102.02751.pdf) [[Website]](https://cvir.github.io/TCL/)

This repository contains the implementation details of our Temporal Contrastive Learning (TCL) approach for action recognition in videos.

Ankit Singh*, Omprakash Chakraborty*, Ashutosh Varshney, Rameswar Panda, Rogerio Feris, Kate Saenko and Abir Das, "Semi-Supervised Action Recognition with Temporal Contrastive Learning"\
*: Equal contributions

If you use the codes and models from this repo, please cite our work. Thanks!

```
@InProceedings{Singh_2021_CVPR,
    author    = {Singh, Ankit and Chakraborty, Omprakash and Varshney, Ashutosh and Panda, Rameswar and Feris, Rogerio and Saenko, Kate and Das, Abir},
    title     = {Semi-Supervised Action Recognition With Temporal Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10389-10399}
}
```

## Requirements
The code is written for python `3.6.10`, but should work for other version with some modifications.
```
pip install -r requirements.txt
```
## Data Preparation

The dataloader (ops/dataset.py) can load videos (image sequences) stored in the following format:
```
-- dataset_dir
---- data
------category.txt  
------train.txt
------val.txt 
---- Frames
------ video_0_folder
-------- 00001.jpg
-------- 00002.jpg
-------- ...
------ video_1_folder
------ ...
```
For each dataset, `root_dataset.yaml` should contain the `dataset_dir` where each dataset is stored


Each line in `train.txt` and `val.txt` includes 3 elements and separated by space. 
Four elements (in order) include (1)relative paths to `video_x_folder` from `dataset_dir`, (2) total number of frames, (3) label id (a numeric number).

E.g., a `video_x` has `300` frames and belong to label `1`.
```
path/to/video_x_folder 300 1
```

After that, in the ops/dataset_config.py, the location paths of `category.txt`, `Frames`, `train.txt` and `val.txt` should be included accordingly.

Samples for some datasets are already mentioned in the respective files.

We provided three sample scripts in the `tools` folder to help convert some datasets but the details in the scripts must be set accordingly. E.g., the path to videos.

## Mini-datasets
We provide the [`category.txt`](datasets/Mini-Something-V2/data/category.txt), [`train.txt`](datasets/Mini-Something-V2/data/train.txt) and [`val.txt`](datasets/Mini-Something-V2/data/val.txt) for the Mini-Something-Something V2 dataset.

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


### Training TCL
- For running `x%` labeled data scenario, it expects to have a folder named `Run_x` where all the labeled and unlabeled data will be split as per the input seed.
- All the models and logs will be stored inside a sub folder of checkpoints directory. A different subfolder will be created on each execution.

### Sample Code to train TCL

`python main.py somethingv2 RGB --seed 123 --strategy classwise
 --arch resnet18 --num_segments 8 --second_segments 4 --threshold 0.8 --gd 20 --lr 0.02 --wd 1e-4 
 --epochs 400 --percentage 0.95 --batch-size 8 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --print-freq 50
 --shift --shift_div=8 --shift_place=blockres --npb --gpus 0 1  --mu 3 --gamma 9 --gamma_finetune 1 
--use_group_contrastive --use_finetuning --finetune_start_epoch 350 --sup_thresh 50 --valbatchsize 16 --finetune_lr 0.002`

## Reference

The implementation reused some portions from [TSM](https://github.com/mit-han-lab/temporal-shift-module)[1].


1. Lin, Ji, Chuang Gan, and Song Han. "Tsm: Temporal shift module for efficient video understanding." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
