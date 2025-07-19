import os
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_weights(model, filename, path="./weight"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, scheduler, epoch, step, filename, root="./checkpoint"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        , fpath)


def load_weights(model, filename):
    state_dict = torch.load(filename)
    if not hasattr(model, 'module'):
        nc = {}
        for k in state_dict:
            nk = k.replace('module.', '')
            nc[nk] = state_dict[k]
    else:
        nc = state_dict
    model.load_state_dict(nc)
    return model


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')
    opt_state = ckpt.get('optimizer', None)
    sch_state = ckpt.get('scheduler', None)
    epoch = ckpt['epoch']
    step = ckpt['step']

    if 'model' in ckpt:
        ckpt = ckpt['model']
        if not hasattr(model, 'module'):
            nc = {}
            for k in ckpt:
                nk = k.replace('module.', '')
                nc[nk] = ckpt[k]
        else:
            nc = ckpt
        model.load_state_dict(nc)
 

    return model, opt_state, sch_state, epoch, step