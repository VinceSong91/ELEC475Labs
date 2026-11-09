"""Compare pretrained teacher, three baseline students, and custom student."""
import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader
from models.student import StudentNet
from models.custom_student import CustomStudentNet
from utils.dataset import get_voc
from utils.metrics import update_confusion, mean_iou
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='./data')
    p.add_argument('--size', type=int, default=320)
    p.add_argument('-b', '--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--examples', type=int, default=4)
    args = p.parse_args()
    kind = ('cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu')) if args.device == 'auto' else args.device
    device = torch.device(kind)
    loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers)
    models = [('Teacher (FCN-ResNet50)', load_teacher(device))]
    for kd, name in [('none', 'Student, no KD'), ('response', 'Student, response KD'),
                     ('feature', 'Student, feature KD')]:
        path = f'checkpoints/student_{kd}.pth'
        if os.path.exists(path):
            m = StudentNet(pretrained=False).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device))
            models.append((name, m))
        else:
            print(f'Skipping {name}: {path} missing')
    for kd in ('response', 'feature', 'none'):
        path = f'checkpoints/student_custom_{kd}.pth'
        if os.path.exists(path):
            m = CustomStudentNet(pretrained=False).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device))
            models.append((f'Custom student, {kd} KD', m))
    os.makedirs('plots', exist_ok=True)
    rows, examples = [], {}
    first_img = first_mask = None
    with torch.no_grad():
        for name, model in models:
            conf = torch.zeros(21, 21, dtype=torch.long)
            for i, (img, mask) in enumerate(loader):
                pred = model(img.to(device))['out'].argmax(1).cpu()
                update_confusion(conf, pred, mask)
                if i == 0:
                    n = min(args.examples, len(img))
                    examples[name] = pred[:n]
                    first_img, first_mask = img[:n], mask[:n]
            params = sum(p.numel() for p in model.parameters())
            ms = ms_per_image(model, device, args.size)
            rows.append({'Model': name, 'mIoU': round(100 * mean_iou(conf), 3),
                         'Params': params, 'ms/img': round(ms, 3)})
            print(f'{name:<30} {rows[-1]["mIoU"]:>7.2f}% '
                  f'{params:>12,} {ms:>9.2f} ms/img', flush=True)
    with open('plots/comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'mIoU', 'Params', 'ms/img'])
        writer.writeheader()
        writer.writerows(rows)
    if first_img is not None:
        save_comparison(first_img, first_mask, examples, 'plots/compare.png')
        print('Saved plots/compare.png and plots/comparison.csv')


if __name__ == '__main__':
    main()
