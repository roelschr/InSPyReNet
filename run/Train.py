import os
import cv2
import sys
import torch
import datetime

import torch.nn as nn
import torch.distributed as dist
import torch.cuda as cuda
import ray

from ray.experimental import tqdm_ray as tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from lib.optim import *
from data.dataloader import *
from utils.misc import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def train(config):
    opt = config["opt"]
    args = config["args"]

    # dataset
    print("HERE: ", os.listdir(opt.Train.Dataset.root), os.path.abspath(os.curdir))
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, 
        sets=opt.Train.Dataset.sets,
        tfs=opt.Train.Dataset.transforms)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=opt.Train.Dataloader.batch_size,
                            num_workers=opt.Train.Dataloader.num_workers,
                            pin_memory=opt.Train.Dataloader.pin_memory,
                            drop_last=True)

    ray.train.torch.prepare_data_loader(train_loader)    

    # model
    model_ckpt = None
    state_ckpt = None
    
    if args.resume is True:
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth')):
            model_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from checkpoint')
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'state.pth')):
            state_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from state')
        
    model = eval(opt.Model.name)(**opt.Model)
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt)

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    
    optimizer = eval(opt.Train.Optimizer.type)(
        params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    
    if state_ckpt is not None:
        optimizer.load_state_dict(state_ckpt['optimizer'])
    
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                                minimum_lr=opt.Train.Scheduler.minimum_lr,
                                                max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,
                                                warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    if state_ckpt is not None:
        scheduler.load_state_dict(state_ckpt['scheduler'])

    model.train()
    model = ray.train.torch.prepare_model(model)

    start = 1
    if state_ckpt is not None:
        start = state_ckpt['epoch']
        
    epoch_iter = range(start, opt.Train.Scheduler.epoch + 1)
    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(epoch_iter, desc='Epoch', total=opt.Train.Scheduler.epoch,
                                position=0)

    for epoch in epoch_iter:
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1)

        else:
            step_iter = enumerate(train_loader, start=1)
        loss = -1
        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True and scaler is not None:
                out = model(sample)
                scaler.scale(out['loss']).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                out = model(sample)
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            loss = out['loss'].item()
            if args.local_rank <= 0 and args.verbose is True:
                tqdm.safe_print({'loss': loss})

        if args.local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                if args.device_num > 1:
                    model_ckpt = model.module.state_dict()  
                else:
                    model_ckpt = model.state_dict()
                    
                state_ckpt = {'epoch': epoch + 1,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}
                
                torch.save(model_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))
                torch.save(state_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'))

                metrics = {"loss": loss, "epoch": epoch}
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(opt.Train.Checkpoint.checkpoint_dir),
                )
            if args.debug is True:
                debout = debug_tile(sum([out[k] for k in opt.Train.Debug.keys], []), activation=torch.sigmoid)
                cv2.imwrite(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)
    
    if args.local_rank <= 0:
        torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    train(opt, args)
