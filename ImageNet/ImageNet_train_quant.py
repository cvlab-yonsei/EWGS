# ref. https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter
import logging
from custom_models import *
from custom_modules import QConv
from utils import update_grad_scales

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# data and model
parser = argparse.ArgumentParser(description='PyTorch Implementation of EWGS')
parser.add_argument('--data', metavar='DIR', default='/path/to/ILSVRC2012', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_quant', choices=('resnet18_quant','resnet34_quant'),
                    help='model architecture')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers')


# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer_m', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--lr_m', type=float, default=1e-2, help='learning rate for model parameters')
parser.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
parser.add_argument('--lr_m_end', type=float, default=0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--lr_q_end', type=float, default=0, help='final learning rate for quantizer parameters (for cosine)')
parser.add_argument('--decay_schedule', type=str, default='40-80', help='learning rate decaying schedule (for step)')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=str2bool)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--pretrained', dest='pretrained', type=str2bool, default=True, help='use pre-trained model')

# misc & distributed data parallel
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# arguments for quantization
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scsaling factor using Hessian trace')
parser.add_argument('--update_scales_every', type=int, default=1, help='update interval in terms of epochs')
parser.add_argument('--visible_gpus', default=None, type=str, help='total GPUs to use')

# logging
parser.add_argument('--log_dir', type=str, default='../results/ResNet18/ours(hess)/W1A1/')

best_acc1 = 0


def main():
    args = parser.parse_args()
    arg_dict = vars(args)

    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= args.visible_gpus

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
    log_string = 'configs\n'
    for k, v in arg_dict.items():
        log_string += "{}: {}\t".format(k,v)
        print("{}: {}".format(k,v), end='\t')
    logging.info(log_string+'\n')
    print('')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def init_quant_model(model, args, ngpus_per_node):
    for m in model.modules():
        if isinstance(m, QConv):
            m.init.data.fill_(1)

    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch_size / ngpus_per_node), shuffle=True,
        num_workers=int((args.workers + ngpus_per_node - 1) / ngpus_per_node), pin_memory=True)
    iterloader = iter(train_loader)
    images, labels = next(iterloader)

    model.to(args.gpu)
    images = images.to(args.gpu)
    labels = labels.to(args.gpu)

    model.train()
    model.forward(images)
    for m in model.modules():
        if isinstance(m, QConv):
            m.init.data.fill_(0)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model_class = globals().get(args.arch)
    model = model_class(args, pretrained=args.pretrained)

    ### initialze quantizer parameters
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0): # do only at rank=0 process
        init_quant_model(model, args, ngpus_per_node)
    ###

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []
    for m in model.modules():
        if isinstance(m, QConv):
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            if m.quan_weight:
                quant_params.append(m.lW)
                quant_params.append(m.uW)
            if m.quan_act:
                quant_params.append(m.lA)
                quant_params.append(m.uA)
            if m.quan_act or m.quan_weight:
                quant_params.append(m.output_scale)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
        elif isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                model_params.append(m.weight)
                model_params.append(m.bias)
    print("# total params:", sum(p.numel() for p in trainable_params))
    print("# trainable model params:", sum(p.numel() for p in model_params))
    print("# trainable quantizer params:", sum(p.numel() for p in quant_params))
    if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
        raise Exception('Mismatched number of trainable parmas')

    if args.optimizer_m == 'SGD':
        optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer_m == 'Adam':
        optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    if args.lr_scheduler == 'cosine':
        scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)
    elif args.lr_scheduler == 'step':
        if args.decay_schedule is not None:
            milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
        else:
            milestones = [(args.epochs+1)]
        scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones, gamma=args.gamma)
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones, gamma=args.gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            print(model.load_state_dict(checkpoint['state_dict']))
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            optimizer_q.load_state_dict(checkpoint['optimizer_q'])
            scheduler_m.load_state_dict(checkpoint['scheduler_m'])
            scheduler_q.load_state_dict(checkpoint['scheduler_q'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, args, None, args.start_epoch)
        return

    ### tensorboard
    if args.rank % ngpus_per_node == 0: # do only at rank=0 process
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    ###

    for epoch in range(args.start_epoch, args.epochs):
        ### update hess scales
        if not args.baseline and args.use_hessian and epoch % args.update_scales_every == 0 and epoch != 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
                print("updating scaling factors")
                update_grad_scales(args)
            dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(os.path.join(args.log_dir, 'update_scales.pth.tar'), map_location=loc)
            print(model.load_state_dict(checkpoint['state_dict'], strict=False))
            print("scaling factors are updated@gpu{}".format(args.gpu))


        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m, scheduler_q, epoch, args, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, writer, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer_m' : optimizer_m.state_dict(),
                'optimizer_q' : optimizer_q.state_dict(),
                'scheduler_m' : scheduler_m.state_dict(),
                'scheduler_q' : scheduler_q.state_dict(),
            }, is_best, path=args.log_dir)


def train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m, scheduler_q, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        optimizer_q.zero_grad()
        loss.backward()
        optimizer_m.step()
        optimizer_q.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if writer is not None: # this only works at rank=0 process
                writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], len(train_loader)*epoch + i)
                writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], len(train_loader)*epoch + i)
                writer.add_scalar('train/loss(current)', loss.cpu().item(), len(train_loader)*epoch + i)
                writer.add_scalar('train/loss(average)', losses.avg, len(train_loader)*epoch + i)
                writer.add_scalar('train/top1(average)', top1.avg, len(train_loader)*epoch + i)
                writer.add_scalar('train/top5(average)', top5.avg, len(train_loader)*epoch + i)
                num_modules = 1
                for m in model.modules():
                    if isinstance(m, QConv):
                        if m.quan_act:
                            writer.add_scalar('z_{}th_module/lA'.format(num_modules), m.lA.item(), len(train_loader)*epoch + i)
                            writer.add_scalar('z_{}th_module/uA'.format(num_modules), m.uA.item(), len(train_loader)*epoch + i)
                            writer.add_scalar('z_{}th_module/bkwd_scaleA'.format(num_modules), m.bkwd_scaling_factorA.item(), len(train_loader)*epoch + i)
                        if m.quan_weight:
                            writer.add_scalar('z_{}th_module/lW'.format(num_modules), m.lW.item(), len(train_loader)*epoch + i)
                            writer.add_scalar('z_{}th_module/uW'.format(num_modules), m.uW.item(), len(train_loader)*epoch + i)
                            writer.add_scalar('z_{}th_module/bkwd_scaleW'.format(num_modules), m.bkwd_scaling_factorW.item(), len(train_loader)*epoch + i)
                        if m.quan_act or m.quan_weight:
                            writer.add_scalar('z_{}th_module/output_scale'.format(num_modules), m.output_scale.item(), len(train_loader)*epoch + i)
                        num_modules += 1

            progress.display(i)
    scheduler_m.step()
    scheduler_q.step()


def validate(val_loader, model, criterion, args, writer, epoch=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if writer is not None:
            writer.add_scalar('val/top1', top1.avg, epoch)
            writer.add_scalar('val/top5', top5.avg, epoch)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, path='./'):
    torch.save(state, os.path.join(path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(path, 'checkpoint.pth.tar'), os.path.join(path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
