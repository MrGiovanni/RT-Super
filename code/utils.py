import os
import logging
import torch
import torch.distributed as dist
import pdb

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")
                
    
    
def debug_optimizer_group_mismatch(model, optimizer, ckpt, max_print=200):
    """
    ckpt: checkpoint dict loaded by torch.load(), must contain:
      - ckpt["optimizer_state_dict"]
      - ckpt["model_state_dict"] (or adjust key names accordingly)
    Prints:
      - group sizes current vs checkpoint
      - exact EXTRA param NAMES in current group(s) vs checkpoint model
      - exact MISSING param NAMES (present in ckpt model but not in current group)
    """
    ckpt_optim_state = ckpt["optimizer_state_dict"]
    ckpt_model_sd = ckpt.get("model_state_dict", None)

    # Map parameter object -> name
    param_to_name = {p: n for n, p in model.named_parameters()}

    cur_groups = optimizer.param_groups
    cur_sizes = [len(g["params"]) for g in cur_groups]

    ckpt_groups = ckpt_optim_state.get("param_groups", [])
    ckpt_sizes = [len(g["params"]) for g in ckpt_groups]

    print("\n[OPT DEBUG] ===== Optimizer param_group size check =====")
    print("[OPT DEBUG] current group sizes   :", cur_sizes)
    print("[OPT DEBUG] checkpoint group sizes:", ckpt_sizes)

    n_total = sum(1 for _ in model.parameters())
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[OPT DEBUG] model params: total={n_total}, trainable={n_trainable}")

    # Helper: strip DDP prefix to compare to ckpt model_state_dict keys
    def norm_name(n: str) -> str:
        return n[7:] if n.startswith("module.") else n

    # If we have ckpt model sd, we can compute EXACT extra/missing NAMES
    ckpt_keys = set(ckpt_model_sd.keys()) if ckpt_model_sd is not None else None

    n = max(len(cur_groups), len(ckpt_groups))
    for i in range(n):
        cur_len = cur_sizes[i] if i < len(cur_sizes) else None
        ckpt_len = ckpt_sizes[i] if i < len(ckpt_sizes) else None
        if cur_len != ckpt_len:
            print(f"\n[OPT DEBUG] --- Group {i} mismatch: current={cur_len} vs ckpt={ckpt_len} ---")

            if i >= len(cur_groups):
                print("[OPT DEBUG] Current optimizer has no such group index.")
                continue

            # Current group i names
            cur_names = []
            for p in cur_groups[i]["params"]:
                cur_names.append(param_to_name.get(p, f"<unnamed_param_{id(p)}>"))

            cur_names_sorted = sorted(cur_names)
            print(f"[OPT DEBUG] Current group {i} param names (sorted, {len(cur_names_sorted)}):")
            for j, name in enumerate(cur_names_sorted[:max_print]):
                print(f"  {j:4d}: {name}")
            if len(cur_names_sorted) > max_print:
                print(f"  ... {len(cur_names_sorted)-max_print} more")

            # EXACT "extra" vs ckpt model keys (this is what you want)
            if ckpt_keys is not None:
                cur_norm = [norm_name(n) for n in cur_names if not n.startswith("<unnamed_param_")]
                cur_norm_set = set(cur_norm)

                # params in current group but not in ckpt model => NEW params
                extra = sorted([n for n in cur_norm_set if n not in ckpt_keys])

                # params that ckpt model had but are missing from current group
                # (restricted to params that still exist in current model, otherwise it's a different issue)
                model_keys_now = set(norm_name(k) for k in model.state_dict().keys())
                missing = sorted([k for k in ckpt_keys if k in model_keys_now and k not in cur_norm_set])

                print(f"\n[OPT DEBUG] >>> EXTRA params in CURRENT group {i} not found in CKPT model_state_dict: {len(extra)}")
                for n in extra[:max_print]:
                    print("   +", n)
                if len(extra) > max_print:
                    print(f"   ... {len(extra)-max_print} more")

                # This can be huge; only print if small
                if len(missing) <= 50:
                    print(f"\n[OPT DEBUG] >>> MISSING params (in CKPT model & current model, but not in CURRENT group {i}): {len(missing)}")
                    for n in missing:
                        print("   -", n)
                else:
                    print(f"\n[OPT DEBUG] >>> MISSING params count is large: {len(missing)} (not printing)")
            else:
                print("[OPT DEBUG] No ckpt['model_state_dict'] found; cannot compute exact extra/missing names.")

    print("\n[OPT DEBUG] ===== End =====\n")
            
def resume_load_optimizer_checkpoint(optimizer, args, model=None):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)

    
    try:    
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except ValueError as e:
        print(f"[OPT DEBUG] load_state_dict failed: {e}")
        if model is not None:
            debug_optimizer_group_mismatch(
                model,
                optimizer,
                checkpoint,
            )
        raise
    

def resume_load_model_checkpoint(net, ema_net, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    args.start_epoch = checkpoint['epoch']

    if args.ema:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)

    print('Loaded checkpoint from (resume):', args.load)



class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" 


def is_master(args):
    return args.rank % args.ngpus_per_node == 0
