import argparse
parser = argparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--labeled_train_list', type=str, default="")
parser.add_argument('--unlabeled_train_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--num_clip', type=int, default=10, help='For number of clips for Video Acc')
parser.add_argument('--num_crop', type=int, default=3, help='For number of crops for Video Acc')
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int,
                    help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default="")
parser.add_argument('--tune_from', type=str, default=None,
                    help='fine-tune from checkpoint')
parser.add_argument('--strategy', type=str, default='classwise', help='[classwise, overall] strategy for sampling')
parser.add_argument('--resume_pretrain',type=str, default='pretrain', help='[finetune, pretrain] which part to resume training ONLY FOR UNS_PRETRAIN')
parser.add_argument('--valbatchsize', default=16, type=int, help='mini-batch size for validation')
parser.add_argument('--sup_thresh',default=50,type=int, help='threshold epchs for pseduo label calculation')
parser.add_argument('--use_group_contrastive',action ='store_true', default=False)
parser.add_argument('--use_finetuning', action ='store_true',default=False)
parser.add_argument('--finetune_start_epoch', default=400, type =int, help='when to start the fine-tune using PL')
parser.add_argument('--Temperature', default=0.5, type=float, help='temperature for sharpening')
parser.add_argument('--finetune_lr', default=-1.0, type=float, help='set fine tune lr for last stage PL')
parser.add_argument('--gamma_finetune',default=9.0, type=float, help= 'weight for pl_loss')
parser.add_argument('--finetune_stage_eval_freq', default=1, type=int, help='frequency for evaluating at finetuning stage')
parser.add_argument('--finetune_epochs', type=int, default=100)
parser.add_argument('--start_finetune_epoch', type=int, default=0)
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cos', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb',
                    default=False, action="store_true")
parser.add_argument('--gamma', default=1.0, type=float,
                    metavar='G', help='weight of contrastive loss')
parser.add_argument('--percentage', default=0.95, type=float,
                    help='should be between 0 and 1. decides percent of training\
     data to be allocated to unlabeled data')
parser.add_argument('--threshold', default=0.95, type=float,
                  help='threshold for pseduo labels')
parser.add_argument('--mu',default=8, type=float, help= 'coefficient for unlabeled data')
parser.add_argument('--second_segments', default=2, type=int, help='no of segments for second branch')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for labeled and unlabeled data separation')
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log', type=str, default='checkpoint')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--shift', default=False,
                    action="store_true", help='use shift for models')
parser.add_argument('--flip', action="store_true", help='Mention this flag if RandomHorizontalFlip is required else do not mention')
parser.add_argument('--shift_div', default=8, type=int,
                    help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres',
                    type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False,
                    action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False,
                    action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False,
                    action="store_true", help='use dense sample for video dataset')
