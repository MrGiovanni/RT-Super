import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
import training.losses_foundation as lf
import math


def get_optimizer(args, net):
    """
    Build an optimizer with separate LR / WD for:
        • core model               – args.base_lr
        • loss_wrapper parameters  – args.base_lr  (× wrapper_lr_mult)
        • att_classifier head      – args.base_lr  (× head_lr_mult)
        • tumor_classifier head    – args.base_lr  (× head_lr_mult)
    """

    # Helper ───────────────────────────────────────────────────────────
    def module_has_attr(module, attr):
        return getattr(module, attr, None) is not None

    # Pick .module if DataParallel / DDP else plain net
    root = net.module if hasattr(net, "module") else net

    wrapper_lr_mult = 6e-4
    head_lr_mult    = 1e-3 #args.base_lr * (1/args.report_volume_loss_basic)

    named_params = dict(root.named_parameters())

    # Split by substring in **name**
    loss_wrapper_params = [p for n, p in named_params.items() if "loss_wrapper"   in n]
    att_params          = [p for n, p in named_params.items() if "att_classifier" in n]
    registration_params = [p for n, p in named_params.items() if "regis_net" in n]
    tumor_cls_params    = [p for n, p in named_params.items() if "tumor_classifier" in n]

    # Everything else that still requires grad
    special_ids = {id(p) for p in loss_wrapper_params + att_params + tumor_cls_params + registration_params}
    model_params = [p for p in root.parameters() if id(p) not in special_ids]

    param_groups = [
        {"params": model_params,
         "lr": args.base_lr,
         "weight_decay": args.weight_decay},
    ]

    if loss_wrapper_params:
        param_groups.append(
            {"params": loss_wrapper_params,
             "lr": wrapper_lr_mult,
             "weight_decay": 0.0}
        )
        
    if registration_params:
        param_groups.append(
            {"params": registration_params,
             "lr": args.base_lr/2,
             "weight_decay": 0.0}
        )

    if att_params:
        param_groups.append(
            {"params": att_params,
             "lr": head_lr_mult,
             "weight_decay": args.weight_decay}
        )
    else:
        if args.attenuation_classifier != 'none' and args.attenuation_classifier != 'simple':
            raise ValueError("Attenuation classifier requested but not found in model!")

    if tumor_cls_params:
        param_groups.append(
            {"params": tumor_cls_params,
             "lr": head_lr_mult,
             "weight_decay": args.weight_decay}
        )
    else:
        if args.tumor_classifier:
            raise ValueError("Tumor classifier requested but not found in model!")
        

    # Choose optimizer
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups,
                                    lr=args.base_lr,          # master LR still required
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups,
                                     lr=args.base_lr,
                                     betas=args.betas,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups,
                                      lr=args.base_lr,
                                      betas=args.betas,
                                      weight_decay=args.weight_decay,
                                      eps=1e-5)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return optimizer


def log_evaluation_result(writer, dice_list, ASD_list, HD_list, name, epoch, args):
    C = dice_list.shape[0]

    writer.add_scalar('Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
    writer.add_scalar('ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    writer.add_scalar('HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)

def unwrap_model_checkpoint(net, ema_net, args):
    net_state_dict = net.module if args.distributed else net 
    net_state_dict = net_state_dict._orig_mod.state_dict() if args.torch_compile else net_state_dict.state_dict()
    if args.ema:
        if args.distributed:
            ema_net_state_dict = ema_net.module.state_dict()
        else:   
            ema_net_state_dict = ema_net.state_dict()
    else:       
        ema_net_state_dict = None 

    return net_state_dict, ema_net_state_dict

def filter_validation_results(dice_list, ASD_list, HD_list, args):
    if args.dataset == 'amos_mr':
        # the validation set of amos_mr doesn't have the last two organs, so elimiate them
        dice_list, ASD_list, HD_list = dice_list[:-2], ASD_list[:-2], HD_list[:-2]

    return dice_list, ASD_list, HD_list

def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr



def exp_lr_scheduler_with_warmup(optimizer,
                                 epoch,
                                 warmup_epoch,
                                 max_epoch,
                                 cosine=False):
    """
    • Keeps each param‑group’s *relative* LR spacing
    • Uses your original warm‑up + polynomial decay curve
    """

    # ------------------------------------------------------------------
    # remember the initial LR of every group (only the first time)
    # ------------------------------------------------------------------
    for g in optimizer.param_groups:
        g.setdefault("base_lr", g["lr"])

    # ------------------------------------------------------------------
    # compute global multiplier
    # ------------------------------------------------------------------
    if warmup_epoch and 0 <= epoch <= warmup_epoch:
        lr_mult = math.exp(10.0 * (float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:          # exact 1.0 at boundary
            lr_mult = 1.0
    else:
        lr_mult = (1.0 - epoch / max_epoch) ** 0.9

    # ------------------------------------------------------------------
    # apply to every group
    # ------------------------------------------------------------------
    i=0
    for g in optimizer.param_groups:
        if i==0:
            g["lr"] =g["base_lr"] * lr_mult
        else:
            if (warmup_epoch and 0 <= epoch <= warmup_epoch):
                g["lr"] = g["base_lr"]
            else:
                g["lr"] = g["base_lr"] * lr_mult
        i+=1

    # optional: return the first group's LR for logging
    return optimizer.param_groups[0]["lr"]



def update_ema_variables(model, ema_model, alpha, global_step):
    
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(m_buffer)



@torch.no_grad()
def concat_all_gather(tensor):
    """ 
    Performs all_gather operation on the provided tensor
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
    """ 
    Due to the DistributedSampler will pad samples for evenly distribute
    samples to gpus, the padded samples need to be removed for right
    evaluation. Need to turn shuffle to False for the dataloader.
    """
    if ranks == 0:
        return tensor

    world_size = dist.get_world_size()
    single_length = len(tensor) // world_size
    output = []

    for rank in range(world_size):
        sub_tensor = tensor[rank * single_length : (rank+1) * single_length]
        if rank >= ranks:
            output.append(sub_tensor[:-1])
        else:
            output.append(sub_tensor)

    output = torch.cat(output)

    return output

