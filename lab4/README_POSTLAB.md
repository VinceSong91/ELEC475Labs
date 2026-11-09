# ELEC 475 Lab 4 post-lab

This project extends the uploaded in-lab source. Run commands from the `lab4` directory.

## Setup

Activate your Python virtual environment and install `pip install -r requirements.txt`.
Use the existing PASCAL VOC 2012 dataset under `./data` (not included in the ZIP).
Run `python test_kd.py` before training. Python 3.9 or later is supported.

## Training

See `train.txt`. All baseline runs use the same batch size, optimizer, LR, scheduler, and early stopping settings. The maximum of 100 epochs is a cap, not a fixed training duration. The best validation-mIoU checkpoint is saved after each improved epoch. Training histories are stored as JSON next to checkpoints. For macOS, `--workers 0` avoids spawn errors, and `--device auto` chooses MPS if available.

## Custom architecture

`models/custom_student.py` uses a MobileNetV3-Small pretrained encoder (same low/mid/high taps as the starter), a custom gated depthwise multiscale context block (dilations 1/3/6), and a dual-skip lightweight decoder. Its final 21-class logits are bilinearly upsampled to the input size. This is a custom head, not a copy of the baseline ASPP head. Teacher and feature adapters are training-only. `--model custom` selects this model.

## Evaluation

See `test.txt`. `evaluate.py` loads the teacher and whichever trained checkpoints exist, evaluates all VOC2012 validation images, prints mIoU/parameters/inference latency, and writes `plots/compare.png` and `plots/comparison.csv`. Missing checkpoints are skipped, not assigned invented scores.

## Report

`Lab4_PostLab_Report_DRAFT.pdf` is a draft with known architecture and methodology, plus placeholders for training and evaluation results. After training, use checkpoint history JSON files and comparison.csv to replace the placeholders and insert actual loss plots and segmentation examples before submission. Include team names/student numbers.

## Submission

The instructor requires a single `<student_number>.zip` with all source code, train.txt, test.txt, trained checkpoints, plots, and final PDF report. Do not include `.venv`, VOC data, or `__pycache__`.
