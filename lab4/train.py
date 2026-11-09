"""Lab 4 full training: best-mIoU checkpoints, plateau LR scheduler, early stopping."""
import argparse
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.student import StudentNet
from models.custom_student import CustomStudentNet
from utils.dataset import get_voc
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['original', 'custom'], default='original')
    p.add_argument('--kd', choices=['none', 'response', 'feature'], default='none')
    p.add_argument('--data', default='./data')
    p.add_argument('-e', '--epochs', type=int, default=100)
    p.add_argument('-b', '--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--tau', type=float, default=4.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--tol', type=float, default=0.1, help='mIoU improvement in percentage points')
    p.add_argument('--iters', type=int, default=0, help='short smoke test; no checkpoint')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = p.parse_args()
    torch.manual_seed(0)
    if args.device == 'auto':
        kind = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        kind = args.device
    device = torch.device(kind)
    train_loader = DataLoader(get_voc(args.data, 'train', args.size), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers)
    student = (CustomStudentNet() if args.model == 'custom' else StudentNet()).to(device)
    teacher = load_teacher(device) if args.kd != 'none' else None
    adapters = None
    params = list(student.parameters())
    if args.kd == 'feature':
        adapters = nn.ModuleDict({'low': nn.Conv2d(24, 512, 1),
                                  'high': nn.Conv2d(576, 2048, 1)}).to(device)
        params += list(adapters.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.lr_factor, patience=args.lr_patience)
    prefix = 'student_' + ('' if args.model == 'original' else 'custom_') + args.kd
    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print(f'Training {prefix} on {device}, batch={args.batch}, max_epochs={args.epochs}', flush=True)
    losses, history = [], []
    best_miou, best_epoch, last_improve_epoch = -float('inf'), 0, 0
    start = time.perf_counter()
    significant_best = -float('inf')
    step = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        if adapters is not None:
            adapters.train()
        epoch_losses = []
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            out = student(img)
            if args.kd == 'none':
                loss = F.cross_entropy(out['out'], mask, ignore_index=255)
            else:
                with torch.no_grad():
                    z_t, f_t = teacher_forward(teacher, img)
                if args.kd == 'response':
                    loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta, args.tau)
                else:
                    ce = F.cross_entropy(out['out'], mask, ignore_index=255)
                    low = feature_kd_loss(adapters['low'](out['low']), f_t['layer2'])
                    high = feature_kd_loss(adapters['high'](out['high']), f_t['layer4'])
                    loss = ce + args.gamma * (low + high)
            opt.zero_grad()
            loss.backward()
            opt.step()
            value = float(loss.item())
            losses.append(value)
            epoch_losses.append(value)
            step += 1
            if args.iters and step % 5 == 0:
                print(f'iter {step}/{args.iters} loss {value:.3f}', flush=True)
            if args.iters and step >= args.iters:
                break
        if args.iters:
            if step >= args.iters:
                print(f'Kick-start: first5={sum(losses[:5])/min(5,len(losses)):.3f} '
                      f'last5={sum(losses[-5:])/min(5,len(losses)):.3f}')
                break
            continue
        student.eval()
        conf = torch.zeros(21, 21, dtype=torch.long)
        with torch.no_grad():
            for img, mask in val_loader:
                pred = student(img.to(device))['out'].argmax(1).cpu()
                update_confusion(conf, pred, mask)
        score = 100 * mean_iou(conf)
        lr_before = opt.param_groups[0]['lr']
        scheduler.step(score)
        elapsed = time.perf_counter() - start
        history.append({'epoch': epoch, 'train_loss': sum(epoch_losses)/len(epoch_losses),
                        'val_miou_percent': score, 'lr': lr_before, 'elapsed_seconds': elapsed})
        # Save best observed checkpoint even if gain is smaller than tolerance.
        if score > best_miou:
            best_miou, best_epoch = score, epoch
            torch.save(student.state_dict(), f'checkpoints/{prefix}.pth')
        # Tolerance controls the early-stop counter relative to last meaningful gain.
        if score > significant_best + args.tol:
            significant_best = score
            last_improve_epoch = epoch
        print(f'epoch {epoch}/{args.epochs} loss {history[-1]["train_loss"]:.3f} '
              f'val mIoU {score:.2f} lr {lr_before:.2g} elapsed {elapsed:.0f}s '
              f'best {best_miou:.2f} (epoch {best_epoch})', flush=True)
        with open(f'checkpoints/{prefix}_history.json', 'w') as f:
            json.dump({'config': vars(args), 'device': str(device), 'history': history,
                       'best_miou_percent': best_miou, 'best_epoch': best_epoch}, f, indent=2)
        if epoch - last_improve_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break
    plt.figure(figsize=(7, 4))
    plt.plot(losses, label='training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training loss: {prefix}')
    plt.tight_layout()
    plt.savefig(f'plots/loss_{prefix[len("student_"):]}.png', dpi=140)
    plt.close()
    print(f'Saved plots/loss_{prefix[len("student_"):]}.png')
    if not args.iters:
        print(f'Best mIoU {best_miou:.2f}% at epoch {best_epoch}; elapsed {time.perf_counter()-start:.0f}s')


if __name__ == '__main__':
    main()
