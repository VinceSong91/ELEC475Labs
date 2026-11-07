
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.student import StudentNet
from utils.dataset import get_voc
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--kd',
        choices=['none', 'response', 'feature'],
        default='none'
    )
    parser.add_argument('--data', default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=4.0)
    parser.add_argument('--gamma', type=float, default=1.0)

    # Set to 0 to avoid macOS multiprocessing errors
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument(
        '--iters',
        type=int,
        default=0,
        help='Kick-start: stop after this many iterations (0 = full training)'
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    # Select GPU acceleration when available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load datasets
    train_loader = DataLoader(
        get_voc(args.data, 'train', args.size),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    val_loader = DataLoader(
        get_voc(args.data, 'val', args.size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers
    )

    # Initialize student model
    student = StudentNet().to(device)
    params = list(student.parameters())

    # Load teacher only when knowledge distillation is enabled
    teacher = load_teacher(device) if args.kd != 'none' else None

    # Feature adapters used during training only
    adapters = None

    if args.kd == 'feature':
        adapters = nn.ModuleDict({
            'low': nn.Conv2d(24, 512, kernel_size=1),
            'high': nn.Conv2d(576, 2048, kernel_size=1)
        }).to(device)

        params += list(adapters.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    print(
        f'Training student with kd={args.kd} on {device}' +
        (
            f' for {args.iters} iterations'
            if args.iters
            else f' for {args.epochs} epochs'
        )
    )

    losses = []
    start = time.time()

    # Training loop
    max_epochs = 10**6 if args.iters else args.epochs

    for epoch in range(1, max_epochs + 1):
        student.train()

        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)

            out = student(img)

            if args.kd == 'none':
                # Standard segmentation cross-entropy
                loss = F.cross_entropy(
                    out['out'],
                    mask,
                    ignore_index=255
                )

            else:
                # Teacher is frozen during distillation
                with torch.no_grad():
                    z_t, f_t = teacher_forward(teacher, img)

                if args.kd == 'response':
                    # Response-based knowledge distillation
                    loss = response_kd_loss(
                        out['out'],
                        z_t,
                        mask,
                        args.alpha,
                        args.beta,
                        args.tau
                    )

                else:
                    # Feature-based knowledge distillation
                    ce_loss = F.cross_entropy(
                        out['out'],
                        mask,
                        ignore_index=255
                    )

                    low_loss = feature_kd_loss(
                        adapters['low'](out['low']),
                        f_t['layer2']
                    )

                    high_loss = feature_kd_loss(
                        adapters['high'](out['high']),
                        f_t['layer4']
                    )

                    loss = ce_loss + args.gamma * (
                        low_loss + high_loss
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Stop early for in-lab kick-start testing
            if args.iters:
                if len(losses) % 5 == 0:
                    print(
                        f'iter {len(losses):3d}/{args.iters} '
                        f'loss {loss.item():.3f}'
                    )

                if len(losses) >= args.iters:
                    break

        if args.iters:
            if len(losses) >= args.iters:
                break
            continue

        # Validation after each full epoch
        student.eval()
        conf = torch.zeros(21, 21, dtype=torch.long)

        with torch.no_grad():
            for img, mask in val_loader:
                pred = student(
                    img.to(device)
                )['out'].argmax(dim=1).cpu()

                conf = update_confusion(conf, pred, mask)

        epoch_losses = losses[-len(train_loader):]
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(
            f'epoch {epoch}/{args.epochs} '
            f'loss {avg_loss:.3f} '
            f'val mIoU {100 * mean_iou(conf):.1f} '
            f'({time.time() - start:.0f} s)'
        )

    # Save training results
    os.makedirs('plots', exist_ok=True)

    if args.iters:
        first = sum(losses[:5]) / 5
        last = sum(losses[-5:]) / 5

        print(
            f'Kick-start finished: average loss '
            f'{first:.3f} (first 5) -> '
            f'{last:.3f} (last 5)'
        )

    else:
        os.makedirs('checkpoints', exist_ok=True)

        checkpoint_path = f'checkpoints/student_{args.kd}.pth'

        torch.save(
            student.state_dict(),
            checkpoint_path
        )

        print(f'Saved {checkpoint_path}')

    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(losses)

    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(f'Training loss (kd={args.kd})')

    plt.tight_layout()

    plot_path = f'plots/loss_{args.kd}.png'

    plt.savefig(plot_path, dpi=100)
    plt.close()

    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
