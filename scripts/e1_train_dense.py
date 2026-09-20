import os
import csv
import time
import random
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import TimesformerForVideoClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# UCF101 Dataset
# ---------------------------------------------------------
class UCF101Dataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        class_file,
        num_frames=8,
        image_size=224,
        train=False,
    ):
        self.root = root
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train

        self.class_to_idx = {}

        with open(class_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                self.class_to_idx[name] = int(idx) - 1

        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                rel_path = parts[0]

                if len(parts) > 1:
                    label = int(parts[1]) - 1
                else:
                    class_name = rel_path.split("/")[0]
                    label = self.class_to_idx[class_name]

                full_path = os.path.join(
                    root,
                    rel_path.replace("/", os.sep)
                )

                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def _read_video(self, path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if length <= 0:
            cap.release()
            raise RuntimeError(f"No frames found: {path}")

        indices = np.linspace(
            0,
            max(length - 1, 0),
            self.num_frames
        ).astype(int)

        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros(
                        (self.image_size, self.image_size, 3),
                        dtype=np.uint8
                    )
            else:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

            frames.append(frame)

        cap.release()
        return frames

    def _transform(self, frames):
        output = []

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        do_flip = self.train and random.random() < 0.5

        for frame in frames:
            x = torch.from_numpy(frame)
            x = x.permute(2, 0, 1).float() / 255.0

            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            if do_flip:
                x = torch.flip(x, dims=[2])

            x = (x - mean) / std
            output.append(x)

        return torch.stack(output, dim=0)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        frames = self._read_video(path)
        pixel_values = self._transform(frames)

        return pixel_values, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------
# Freeze Backbone
# ---------------------------------------------------------
def configure_partial_finetuning(model, last_blocks=2):

    for p in model.parameters():
        p.requires_grad = False

    # classifier always trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    # unfreeze final transformer blocks
    layers = model.timesformer.encoder.layer
    n_layers = len(layers)

    start_idx = max(0, n_layers - last_blocks)

    for i in range(start_idx, n_layers):
        for p in layers[i].parameters():
            p.requires_grad = True

    print(f"Transformer layers : {n_layers}")
    print(
        f"Trainable layers   : "
        f"{start_idx} to {n_layers - 1}"
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    y_true = []
    y_pred = []

    total_loss = 0.0
    total_seen = 0

    for videos, labels in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):

        videos = videos.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            pixel_values=videos
        ).logits

        loss = criterion(logits, labels)

        bs = labels.size(0)

        total_loss += loss.item() * bs
        total_seen += bs

        preds = logits.argmax(dim=1)

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

    avg_loss = total_loss / total_seen

    acc = accuracy_score(
        y_true,
        y_pred
    ) * 100.0

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    ) * 100.0

    return avg_loss, acc, f1


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_list = os.path.join(
        args.split_root,
        "trainlist01.txt"
    )

    test_list = os.path.join(
        args.split_root,
        "testlist01.txt"
    )

    class_file = os.path.join(
        args.split_root,
        "classInd.txt"
    )

    # -----------------------------------------------------
    # Separate train and validation dataset objects
    # -----------------------------------------------------
    train_base = UCF101Dataset(
        args.data_root,
        train_list,
        class_file,
        num_frames=args.frames,
        image_size=args.image_size,
        train=True
    )

    val_base = UCF101Dataset(
        args.data_root,
        train_list,
        class_file,
        num_frames=args.frames,
        image_size=args.image_size,
        train=False
    )

    test_dataset = UCF101Dataset(
        args.data_root,
        test_list,
        class_file,
        num_frames=args.frames,
        image_size=args.image_size,
        train=False
    )

    indices = np.arange(len(train_base))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    n_val = int(
        len(indices) * args.val_fraction
    )

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = Subset(
        train_base,
        train_indices
    )

    val_dataset = Subset(
        val_base,
        val_indices
    )

    print()
    print("UCF101 Split-1")
    print("Train :", len(train_dataset))
    print("Val   :", len(val_dataset))
    print("Test  :", len(test_dataset))
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0
    )

    print("Loading TimeSformer...")

    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=101,
        ignore_mismatched_sizes=True
    )

    configure_partial_finetuning(
        model,
        last_blocks=args.last_blocks
    )

    model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params / 1e6:.2f} M"
    )

    print(
        f"Trainable ratio      : "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda"
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        args.results_dir,
        exist_ok=True
    )

    best_acc = -1.0
    no_improve = 0

    history = []

    # -----------------------------------------------------
    # Epochs
    # -----------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0.0
        seen = 0

        start_time = time.time()

        bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}"
        )

        for batch_idx, (videos, labels) in enumerate(bar):

            if (
                args.max_train_batches > 0
                and batch_idx >= args.max_train_batches
            ):
                break

            videos = videos.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda"
            ):

                logits = model(
                    pixel_values=videos
                ).logits

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)

            running_loss += (
                loss.item() * bs
            )

            seen += bs

            bar.set_postfix(
                loss=running_loss / seen
            )

        train_loss = (
            running_loss / max(seen, 1)
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            device
        )

        elapsed = (
            time.time() - start_time
        )

        print()
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.2f}% | "
            f"Val F1 {val_f1:.2f}% | "
            f"{elapsed / 60:.1f} min"
        )

        history.append([
            epoch,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            elapsed
        ])

        if val_acc > best_acc:

            best_acc = val_acc
            no_improve = 0

            ckpt_path = os.path.join(
                args.checkpoint_dir,
                "e1_timesformer_ucf101_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "args": vars(args)
                },
                ckpt_path
            )

            print(
                "Saved best checkpoint:",
                ckpt_path
            )

        else:

            no_improve += 1

            print(
                f"No improvement: "
                f"{no_improve}/{args.patience}"
            )

            if no_improve >= args.patience:

                print(
                    "Early stopping triggered."
                )

                break

    # -----------------------------------------------------
    # Full evaluation skipped in smoke mode
    # -----------------------------------------------------
    if args.smoke_test:
        print()
        print("==============================")
        print("SMOKE TEST COMPLETED")
        print("==============================")
        print(
            "Training pipeline is working."
        )
        return

    # -----------------------------------------------------
    # Reload best model
    # -----------------------------------------------------
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        "e1_timesformer_ucf101_best.pt"
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print()
    print("Evaluating official test set...")

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print("==============================")
    print("E1 FINAL TEST RESULTS")
    print("==============================")
    print(
        f"Top-1 Accuracy : {test_acc:.3f}%"
    )
    print(
        f"Macro-F1       : {test_f1:.3f}%"
    )
    print(
        f"Test Loss      : {test_loss:.4f}"
    )
    print("==============================")

    # -----------------------------------------------------
    # Save history
    # -----------------------------------------------------
    history_file = os.path.join(
        args.results_dir,
        "e1_training_history.csv"
    )

    with open(
        history_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "epoch_seconds"
        ])

        writer.writerows(history)

    # -----------------------------------------------------
    # Save final result
    # -----------------------------------------------------
    result_file = os.path.join(
        args.results_dir,
        "e1_dense_results.csv"
    )

    with open(
        result_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "split",
            "backbone",
            "frames",
            "resolution",
            "total_params_M",
            "trainable_params_M",
            "top1",
            "macro_f1"
        ])

        writer.writerow([
            "UCF101",
            "Split1",
            "TimeSformer-B",
            args.frames,
            args.image_size,
            total_params / 1e6,
            trainable_params / 1e6,
            test_acc,
            test_f1
        ])

    print(
        "Saved:",
        result_file
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--split-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=8
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--last-blocks",
        type=int,
        default=2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results"
    )

    args = parser.parse_args()

    main(args)