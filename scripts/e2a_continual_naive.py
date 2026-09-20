import os
import csv
import random
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TimesformerForVideoClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UCFStageDataset(Dataset):
    def __init__(
        self, root, split_file, class_file,
        allowed_classes, frames=8,
        size=224, train=False
    ):
        self.root = root
        self.frames = frames
        self.size = size
        self.train = train
        self.allowed = set(allowed_classes)

        self.class_to_idx = {}

        with open(class_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                self.class_to_idx[name] = int(idx) - 1

        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                rel = parts[0]

                if len(parts) > 1:
                    label = int(parts[1]) - 1
                else:
                    cname = rel.split("/")[0]
                    label = self.class_to_idx[cname]

                if label in self.allowed:
                    path = os.path.join(
                        root,
                        rel.replace("/", os.sep)
                    )
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def read_video(self, path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")

        length = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if length <= 0:
            cap.release()
            raise RuntimeError(f"No frames: {path}")

        indices = np.linspace(
            0, length - 1, self.frames
        ).astype(int)

        out = []

        for idx in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            ok, frame = cap.read()

            if not ok:
                if out:
                    frame = out[-1].copy()
                else:
                    frame = np.zeros(
                        (self.size, self.size, 3),
                        dtype=np.uint8
                    )
            else:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

            out.append(frame)

        cap.release()
        return out

    def transform(self, frames):
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        flip = (
            self.train and
            random.random() < 0.5
        )

        tensors = []

        for frame in frames:
            x = torch.from_numpy(frame)
            x = (
                x.permute(2, 0, 1)
                .float() / 255.0
            )

            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            if flip:
                x = torch.flip(x, [2])

            x = (x - mean) / std
            tensors.append(x)

        return torch.stack(tensors)

    def __getitem__(self, index):
        path, label = self.samples[index]

        x = self.transform(
            self.read_video(path)
        )

        return x, torch.tensor(
            label,
            dtype=torch.long
        )


def configure_model(model, last_blocks):

    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    layers = model.timesformer.encoder.layer

    start = max(
        0,
        len(layers) - last_blocks
    )

    for idx in range(start, len(layers)):
        for p in layers[idx].parameters():
            p.requires_grad = True

    return start, len(layers)


@torch.no_grad()
def evaluate(
    model, loader, device,
    seen_classes
):
    model.eval()

    truth = []
    pred = []

    mask = torch.ones(
        101,
        dtype=torch.bool,
        device=device
    )

    mask[list(seen_classes)] = False

    for videos, labels in tqdm(
        loader,
        desc="Eval",
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

        # Class-incremental evaluation:
        # future unseen classes cannot be predicted.
        logits[:, mask] = -1e9

        predictions = logits.argmax(1)

        truth.extend(
            labels.cpu().numpy()
        )
        pred.extend(
            predictions.cpu().numpy()
        )

    acc = accuracy_score(
        truth, pred
    ) * 100.0

    f1 = f1_score(
        truth,
        pred,
        labels=sorted(seen_classes),
        average="macro",
        zero_division=0
    ) * 100.0

    return acc, f1


def train_stage(
    model, loader, device,
    epochs, lr, weight_decay,
    amp
):
    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            amp and device.type == "cuda"
        )
    )

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        seen = 0

        bar = tqdm(
            loader,
            desc=f"Train epoch {epoch}/{epochs}"
        )

        for videos, labels in bar:

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
                enabled=(
                    amp and
                    device.type == "cuda"
                )
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

            total_loss += (
                loss.item() * bs
            )
            seen += bs

            bar.set_postfix(
                loss=total_loss / seen
            )


def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("FedCAV E2-A — Naive Continual Video Learning")
    print("=" * 70)
    print("Device :", device)

    if device.type == "cuda":
        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

    class_file = os.path.join(
        args.split_root,
        "classInd.txt"
    )

    train_file = os.path.join(
        args.split_root,
        "trainlist01.txt"
    )

    test_file = os.path.join(
        args.split_root,
        "testlist01.txt"
    )

    # Deterministic 5-stage protocol
    stages = [
        list(range(0, 20)),
        list(range(20, 40)),
        list(range(40, 60)),
        list(range(60, 80)),
        list(range(80, 101))
    ]

    print("\nStages:")
    for i, classes in enumerate(stages, 1):
        print(
            f"S{i}: classes "
            f"{classes[0] + 1}-"
            f"{classes[-1] + 1} "
            f"({len(classes)} classes)"
        )

    print("\nLoading K400-pretrained TimeSformer...")

    model = (
        TimesformerForVideoClassification
        .from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=101,
            ignore_mismatched_sizes=True
        )
    )

    start, total_layers = configure_model(
        model,
        args.last_blocks
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
        f"Transformer layers : {total_layers}"
    )
    print(
        f"Trainable layers   : "
        f"{start}-{total_layers - 1}"
    )
    print(
        f"Total parameters   : "
        f"{total_params / 1e6:.3f} M"
    )
    print(
        f"Trainable params    : "
        f"{trainable_params / 1e6:.3f} M"
    )

    os.makedirs(
        args.results_dir,
        exist_ok=True
    )
    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    # A[t,j] = accuracy on stage-j test data
    # after training through stage t.
    A = np.full(
        (len(stages), len(stages)),
        np.nan
    )

    F1 = np.full_like(
        A,
        np.nan
    )

    for t, current_classes in enumerate(
        stages
    ):

        print("\n" + "=" * 70)
        print(
            f"TRAINING STAGE {t + 1}/5"
        )
        print("=" * 70)

        train_dataset = UCFStageDataset(
            args.data_root,
            train_file,
            class_file,
            current_classes,
            frames=args.frames,
            size=args.image_size,
            train=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(
                args.workers > 0
            )
        )

        print(
            "Training samples:",
            len(train_dataset)
        )

        train_stage(
            model,
            train_loader,
            device,
            args.epochs_per_stage,
            args.lr,
            args.weight_decay,
            args.amp
        )

        # Save each stage model
        ckpt = os.path.join(
            args.checkpoint_dir,
            f"e2a_stage{t + 1}.pt"
        )

        torch.save(
            {
                "stage": t + 1,
                "model": model.state_dict(),
                "classes": current_classes,
                "args": vars(args)
            },
            ckpt
        )

        seen_classes = []

        for j in range(t + 1):

            seen_classes.extend(
                stages[j]
            )

        print(
            f"\nEvaluating {t + 1} "
            f"seen stage(s)..."
        )

        for j in range(t + 1):

            test_dataset = UCFStageDataset(
                args.data_root,
                test_file,
                class_file,
                stages[j],
                frames=args.frames,
                size=args.image_size,
                train=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=(
                    args.workers > 0
                )
            )

            acc, f1 = evaluate(
                model,
                test_loader,
                device,
                seen_classes
            )

            A[t, j] = acc
            F1[t, j] = f1

            print(
                f"After S{t + 1} -> "
                f"Test S{j + 1}: "
                f"Acc={acc:.3f}% "
                f"F1={f1:.3f}%"
            )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------
    final_acc = float(
        np.nanmean(A[-1, :])
    )

    final_f1 = float(
        np.nanmean(F1[-1, :])
    )

    forgetting_values = []

    for j in range(len(stages) - 1):

        best_previous = np.nanmax(
            A[j:, j][:-1]
        )

        final_value = A[-1, j]

        forgetting_values.append(
            best_previous - final_value
        )

    avg_forgetting = float(
        np.mean(forgetting_values)
    )

    # Standard task-wise BWT:
    # final performance minus performance
    # immediately after learning task j.
    bwt_values = []

    for j in range(len(stages) - 1):
        bwt_values.append(
            A[-1, j] - A[j, j]
        )

    bwt = float(
        np.mean(bwt_values)
    )

    print("\n" + "=" * 70)
    print("E2-A FINAL RESULTS")
    print("=" * 70)

    print(
        f"Final Average Accuracy : "
        f"{final_acc:.3f}%"
    )
    print(
        f"Final Average Macro-F1 : "
        f"{final_f1:.3f}%"
    )
    print(
        f"Average Forgetting     : "
        f"{avg_forgetting:.3f} pp"
    )
    print(
        f"Backward Transfer      : "
        f"{bwt:.3f} pp"
    )

    print("\nAccuracy matrix:")
    print(
        np.array2string(
            A,
            precision=3,
            suppress_small=True
        )
    )

    # -----------------------------------------------------
    # Save matrices
    # -----------------------------------------------------
    np.savetxt(
        os.path.join(
            args.results_dir,
            "e2a_accuracy_matrix.csv"
        ),
        A,
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(
            args.results_dir,
            "e2a_f1_matrix.csv"
        ),
        F1,
        delimiter=",",
        fmt="%.6f"
    )

    summary = os.path.join(
        args.results_dir,
        "e2a_summary.csv"
    )

    with open(
        summary,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "final_avg_accuracy",
                "final_avg_macro_f1",
                "avg_forgetting_pp",
                "bwt_pp"
            ]
        )

        writer.writerow(
            [
                final_acc,
                final_f1,
                avg_forgetting,
                bwt
            ]
        )

    print("\nSaved:", summary)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True
    )
    parser.add_argument(
        "--split-root",
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
        "--epochs-per-stage",
        type=int,
        default=1
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
        "--amp",
        action="store_true"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/e2a"
    )
    parser.add_argument(
        "--results-dir",
        default="results"
    )

    args = parser.parse_args()

    main(args)