import torch

def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    elif args.optim.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return 