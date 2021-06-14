import os
import time
import shutil
import random
import datetime
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

best_prec1 = 0


def main():
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    global args, best_prec1
    args = parser.parse_args()

    ##asset check ####
    if args.use_finetuning:
        assert args.finetune_start_epoch > args.sup_thresh

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,args.modality)
    full_arch_name = args.arch
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TCL', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), args.dataset, full_arch_name, 'p%.2f' % args.percentage,'th%.2f' % args.threshold,'gamma%0.2f' % args.gamma,'mu%0.2f'% args.mu,'seed%d' % args.seed,'seg%d' % args.num_segments, 'bs%d' % args.batch_size,
         'e{}'.format(args.epochs)])
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    args.labeled_train_list, args.unlabeled_train_list=get_training_filenames(args.train_list)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                second_segments = args.second_segments,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)
    print("==============model desccription=============")
    print(model)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=args.flip)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    labeled_trainloader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.labeled_train_list, unlabeled=False,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   second_segments = args.second_segments,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(
                           roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(
                           div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=False)  # prevent something not % n_GPU

    unlabeled_trainloader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.unlabeled_train_list, unlabeled=True,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   second_segments = args.second_segments,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(
                           roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(
                           div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=np.int(np.round(args.mu * args.batch_size)), shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=False)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, unlabeled=False,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   second_segments = args.second_segments,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(
                           roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(
                           div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.valbatchsize, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(
        args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    default_start = 0
    is_finetune_lr_set= False
    for epoch in range(args.start_epoch, args.epochs):
        if args.use_finetuning and epoch >= args.finetune_start_epoch:
            args.eval_freq = args.finetune_stage_eval_freq
        if args.use_finetuning and epoch >= args.finetune_start_epoch and args.finetune_lr > 0.0 and not is_finetune_lr_set:
            args.lr = args.finetune_lr
            default_start = args.finetune_start_epoch
            is_finetune_lr_set = True
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps,default_start, using_policy=True)

        # train for one epoch
        train(labeled_trainloader, unlabeled_trainloader, model,
              criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if ((epoch + 1) % args.eval_freq == 0 or epoch == (args.epochs - 1) or (epoch+1)== args.finetune_start_epoch) :
            prec1 = validate(val_loader, model, criterion,
                             epoch, log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()
            if args.use_finetuning and (epoch+1) == args.finetune_start_epoch:
                one_stage_pl=True
            else:
                one_stage_pl = False
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,one_stage_pl)


def train(labeled_trainloader, unlabeled_trainloader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    supervised_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    group_contrastive_losses = AverageMeter()
    pl_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model = model.cuda()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    if epoch >= args.sup_thresh or (args.use_finetuning and epoch >= args.finetune_start_epoch):
        data_loader = zip(labeled_trainloader, unlabeled_trainloader)
    else:
        data_loader = labeled_trainloader

    end = time.time()

    for i, data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #reseting losses
        contrastive_loss = torch.tensor(0.0).cuda()
        pl_loss = torch.tensor(0.0).cuda()
        loss = torch.tensor(0.0).cuda()
        group_contrastive_loss = torch.tensor(0.0).cuda()

        if epoch >= args.sup_thresh  or (args.use_finetuning and epoch >= args.finetune_start_epoch):
            
            (labeled_data,unlabeled_data) =data
            images_fast, images_slow = unlabeled_data
            images_slow = images_slow.cuda()
            images_fast = images_fast.cuda()
            images_slow = torch.autograd.Variable(images_slow)
            images_fast = torch.autograd.Variable(images_fast)

            # contrastive_loss
            output_fast = model(images_fast)
            if not args.use_finetuning or epoch < args.finetune_start_epoch:
                output_slow = model(images_slow, unlabeled=True)
            output_fast_detach = output_fast.detach()
            if epoch >= args.sup_thresh and epoch < args.finetune_start_epoch:
                contrastive_loss = simclr_loss(torch.softmax(output_fast_detach,dim=1),torch.softmax(output_slow,dim=1))
                if args.use_group_contrastive:
                    grp_unlabeled_8seg = get_group(output_fast_detach)
                    grp_unlabeled_4seg = get_group(output_slow)
                    group_contrastive_loss = compute_group_contrastive_loss(grp_unlabeled_8seg,grp_unlabeled_4seg) 
            elif args.use_finetuning  and epoch >= args.finetune_start_epoch:
                pseudo_label = torch.softmax(output_fast_detach, dim=-1)
                max_probs, targets_pl = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                targets_pl = torch.autograd.Variable(targets_pl)

                pl_loss = (F.cross_entropy(output_fast, targets_pl,
                                                    reduction='none') * mask).mean()
        else:
            labeled_data = data
        input, target = labeled_data
        target = target.cuda()
        input = input.cuda()
        input = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input)
        loss = criterion(output, target_var)

        total_loss = loss + args.gamma*contrastive_loss  + group_contrastive_loss + args.gamma_finetune*pl_loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        if epoch >= args.sup_thresh:
            total_losses.update(total_loss.item(), input.size(0)+args.mu*input.size(0))
        else:
            total_losses.update(total_loss.item(), input.size(0))
        supervised_losses.update(loss.item(), input.size(0))
        contrastive_losses.update(contrastive_loss.item(), input.size(0)+args.mu*input.size(0))
        group_contrastive_losses.update(group_contrastive_loss.item(), input.size(0)+args.mu*input.size(0))
        pl_losses.update(pl_loss.item(), input.size(0)+args.mu*input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        total_loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(
                model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'TotalLoss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                      'Supervised Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Contrastive_Loss {contrastive_loss.val:.4f} ({contrastive_loss.avg:.4f})\t'
                      'Group_contrastive_Loss {group_contrastive_loss.val:.4f} ({group_contrastive_loss.avg:.4f})\t'
                      'Pseudo_Loss {pl_loss.val:.4f} ({pl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, batch_time=batch_time,
                          data_time=data_time, total_loss=total_losses,loss=supervised_losses,
                          contrastive_loss=contrastive_losses,group_contrastive_loss=group_contrastive_losses,pl_loss=pl_losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

def validate(val_loader, model, criterion, epoch, log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              i, len(val_loader), batch_time=batch_time, loss=losses,
                              top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    return top1.avg

def get_group(output):
    logits = torch.softmax(output, dim=-1)
    _ , target = torch.max(logits, dim=-1)
    groups ={}
    for x,y in zip(target, logits):
        group = groups.get(x.item(),[])
        group.append(y)
        groups[x.item()]= group
    return groups

def compute_group_contrastive_loss(grp_dict_un,grp_dict_lab):
    loss = []
    l_fast =[]
    l_slow =[]
    for key in grp_dict_un.keys():
        if key in grp_dict_lab:
            l_fast.append(torch.stack(grp_dict_un[key]).mean(dim=0))
            l_slow.append(torch.stack(grp_dict_lab[key]).mean(dim=0))
    if len(l_fast) > 0:
        l_fast = torch.stack(l_fast)
        l_slow = torch.stack(l_slow)
        loss = simclr_loss(l_fast,l_slow)
        loss = max(torch.tensor(0.000).cuda(),loss)
    else:
        loss= torch.tensor(0.0).cuda()
    return loss


def save_checkpoint(state, is_best, one_stage_pl=False):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if one_stage_pl:
        shutil.copyfile(filename, filename.replace('pth.tar', 'before_finetune.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps,default_start,using_policy):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - default_start) / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    if using_policy:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
            param_group['weight_decay'] = decay 
    

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)



def split_file(file, unlabeled, labeled, percentage, isShuffle=True, seed=123, strategy='classwise'):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    if strategy == 'classwise':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        #creating dictionary against each category
        def del_list(list_delete,indices_to_delete):
            for i in sorted(indices_to_delete, reverse=True):
                del(list_delete[i])

        main_dict= defaultdict(list)
        with open(file,'r') as mainfile:
            lines = mainfile.readlines()
            for line in lines:
                video_info = line.strip().split()
                main_dict[video_info[2]].append((video_info[0],video_info[1]))
        with open(unlabeled,'w') as ul,\
            open(labeled,'w') as l:
            for key,value in main_dict.items():
                length_videos = len(value)
                ul_no_videos = int((length_videos* percentage))
                indices = random.sample(range(length_videos),ul_no_videos)
                for index in indices:
                    line_to_written = value[index][0] + " " + value[index][1] + " " +key+"\n"
                    ul.write(line_to_written)
                del_list(value,indices)
                for label_index in range(len(value)):
                    line_to_written = value[label_index][0] + " " + value[label_index][1] + " " +key+"\n"
                    l.write(line_to_written)



    if strategy == 'overall':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        with open(file, 'r') as fin, \
            open(unlabeled, 'w') as foutBig, \
            open(labeled, 'w') as foutSmall:
        # if didn't count you could only approximate the percentage
            lines = fin.readlines()
            random.shuffle(lines)
            nLines = sum(1 for line in lines)
            nTrain = int(nLines*percentage)
            i = 0
            for line in lines:
                line = line.rstrip('\n') + "\n"
                if i < nTrain:
                     foutBig.write(line)
                     i += 1
                else:
                     foutSmall.write(line)

def get_training_filenames(train_file_path):
    labeled_file_path = os.path.join("Run_"+str(int(np.round((1-args.percentage)*100))),args.dataset+'_'+str(args.seed)+args.strategy+"_labeled_training.txt")
    unlabeled_file_path = os.path.join("Run_"+str(int(np.round((1-args.percentage)*100))),args.dataset+'_'+str(args.seed)+args.strategy+"_unlabeled_training.txt")
    split_file(train_file_path, unlabeled_file_path,
               labeled_file_path,args.percentage, isShuffle=True,seed=args.seed, strategy=args.strategy)
    return labeled_file_path, unlabeled_file_path

def simclr_loss(output_fast,output_slow,normalize=True):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    if normalize:
        sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / args.Temperature)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / args.Temperature)
    else:
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / args.Temperature)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / args.Temperature )
    norm_sum = norm_sum.cuda()
    loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
    return loss

if __name__ == '__main__':
    main()
