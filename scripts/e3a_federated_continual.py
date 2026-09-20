import os
import csv
import copy
import random
import argparse
from collections import OrderedDict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import TimesformerForVideoClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# UCF101 Stage Dataset
# ============================================================

class UCFStageDataset(Dataset):

    def __init__(
        self,
        root,
        split_file,
        class_file,
        allowed_classes,
        frames=8,
        size=224,
        train=False
    ):

        self.root = root
        self.frames = frames
        self.size = size
        self.train = train

        self.allowed_classes = set(allowed_classes)

        self.class_to_idx = {}

        with open(class_file, "r", encoding="utf-8") as f:
            for line in f:
                idx, name = line.strip().split()
                self.class_to_idx[name] = int(idx) - 1

        self.samples = []

        with open(split_file, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                rel_path = parts[0]

                if len(parts) > 1:
                    label = int(parts[1]) - 1

                else:

                    class_name = (
                        rel_path
                        .replace("\\", "/")
                        .split("/")[0]
                    )

                    label = self.class_to_idx[
                        class_name
                    ]

                if label not in self.allowed_classes:
                    continue

                full_path = os.path.join(
                    root,
                    rel_path.replace("/", os.sep)
                )

                self.samples.append(
                    (full_path, label)
                )

    def __len__(self):
        return len(self.samples)

    def _read_video(self, path):

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {path}"
            )

        length = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if length <= 0:
            cap.release()
            raise RuntimeError(
                f"No frames found: {path}"
            )

        indices = np.linspace(
            0,
            max(length - 1, 0),
            self.frames
        ).astype(int)

        frames = []

        for frame_idx in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_idx)
            )

            ok, frame = cap.read()

            if not ok:

                if len(frames) > 0:
                    frame = frames[-1].copy()

                else:
                    frame = np.zeros(
                        (
                            self.size,
                            self.size,
                            3
                        ),
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

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        do_flip = (
            self.train
            and random.random() < 0.5
        )

        output = []

        for frame in frames:

            x = torch.from_numpy(frame)

            x = (
                x.permute(2, 0, 1)
                .float()
                / 255.0
            )

            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            if do_flip:
                x = torch.flip(
                    x,
                    dims=[2]
                )

            x = (x - mean) / std

            output.append(x)

        return torch.stack(
            output,
            dim=0
        )

    def __getitem__(self, index):

        path, label = self.samples[index]

        frames = self._read_video(path)

        pixel_values = self._transform(frames)

        return (
            pixel_values,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# Partial Fine-Tuning
# ============================================================

def configure_partial_finetuning(
    model,
    last_blocks=2
):

    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    layers = model.timesformer.encoder.layer

    total_layers = len(layers)

    start_idx = max(
        0,
        total_layers - last_blocks
    )

    for i in range(
        start_idx,
        total_layers
    ):
        for p in layers[i].parameters():
            p.requires_grad = True

    return start_idx, total_layers


# ============================================================
# Dirichlet Non-IID Partition
# ============================================================

def dirichlet_partition(
    dataset,
    num_clients,
    alpha,
    seed
):

    rng = np.random.default_rng(seed)

    labels = np.array(
        [
            int(label)
            for _, label in dataset.samples
        ]
    )

    client_indices = [
        []
        for _ in range(num_clients)
    ]

    unique_classes = np.unique(labels)

    for class_id in unique_classes:

        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.repeat(
                alpha,
                num_clients
            )
        )

        counts = rng.multinomial(
            len(class_indices),
            proportions
        )

        start = 0

        for client_id, count in enumerate(counts):

            if count > 0:

                selected = class_indices[
                    start:start + count
                ]

                client_indices[
                    client_id
                ].extend(
                    selected.tolist()
                )

            start += count

    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices


# ============================================================
# Local Client Training
# ============================================================

def train_local_client(
    model,
    loader,
    device,
    local_epochs,
    lr,
    weight_decay,
    amp,
    max_batches=None
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
            amp
            and device.type == "cuda"
        )
    )

    total_seen = 0
    total_loss = 0.0

    for epoch in range(
        1,
        local_epochs + 1
    ):

        model.train()

        bar = tqdm(
            loader,
            desc=f"Local epoch {epoch}/{local_epochs}",
            leave=False
        )

        for batch_idx, (
            videos,
            labels
        ) in enumerate(bar):

            if (
                max_batches is not None
                and batch_idx >= max_batches
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
                enabled=(
                    amp
                    and device.type == "cuda"
                )
            ):

                logits = model(
                    pixel_values=videos
                ).logits

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            bs = labels.size(0)

            total_seen += bs
            total_loss += (
                loss.item() * bs
            )

            bar.set_postfix(
                loss=(
                    total_loss
                    / max(total_seen, 1)
                )
            )

    return (
        total_loss
        / max(total_seen, 1)
    )


# ============================================================
# Extract Trainable State Only
# ============================================================

def get_trainable_state(model):

    trainable_names = {
        name
        for name, param
        in model.named_parameters()
        if param.requires_grad
    }

    state = OrderedDict()

    for name, tensor in model.state_dict().items():

        if name in trainable_names:

            state[name] = (
                tensor.detach()
                .cpu()
                .float()
                .clone()
            )

    return state


# ============================================================
# FedAvg
# ============================================================

def fedavg(
    client_states,
    client_weights
):

    total_weight = float(
        sum(client_weights)
    )

    averaged = OrderedDict()

    for key in client_states[0].keys():

        avg_tensor = torch.zeros_like(
            client_states[0][key],
            dtype=torch.float32
        )

        for state, weight in zip(
            client_states,
            client_weights
        ):

            avg_tensor += (
                state[key]
                * (
                    float(weight)
                    / total_weight
                )
            )

        averaged[key] = avg_tensor

    return averaged


# ============================================================
# Load Aggregated Trainable State
# ============================================================

def load_trainable_state(
    model,
    averaged_state
):

    current_state = model.state_dict()

    for key, value in averaged_state.items():

        current_state[key] = value.to(
            dtype=current_state[key].dtype
        )

    model.load_state_dict(
        current_state,
        strict=True
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    seen_classes,
    eval_classes,
    amp
):

    model.eval()

    truth = []
    predictions_all = []

    unseen_mask = torch.ones(
        101,
        dtype=torch.bool,
        device=device
    )

    unseen_mask[
        list(seen_classes)
    ] = False

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

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                amp
                and device.type == "cuda"
            )
        ):

            logits = model(
                pixel_values=videos
            ).logits

        logits[:, unseen_mask] = -1e9

        predictions = logits.argmax(
            dim=1
        )

        truth.extend(
            labels.cpu().numpy()
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

    accuracy = (
        accuracy_score(
            truth,
            predictions_all
        )
        * 100.0
    )

    macro_f1 = (
        f1_score(
            truth,
            predictions_all,
            labels=sorted(eval_classes),
            average="macro",
            zero_division=0
        )
        * 100.0
    )

    return accuracy, macro_f1


# ============================================================
# Main
# ============================================================

def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 76)
    print(
        "FedCAV E3-A — "
        "Federated Continual FedAvg Baseline"
    )
    print("=" * 76)

    print("Device            :", device)

    if device.type == "cuda":
        print(
            "GPU               :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Clients           :",
        args.clients
    )

    print(
        "Dirichlet alpha   :",
        args.alpha
    )

    print(
        "Rounds/stage      :",
        args.rounds_per_stage
    )

    print(
        "Local epochs      :",
        args.local_epochs
    )

    print(
        "Last blocks       :",
        args.last_blocks
    )

    if args.smoke_test:
        print(
            "Mode              : SMOKE TEST"
        )

    # --------------------------------------------------------
    # Split files
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Continual stages
    # --------------------------------------------------------

    all_stages = [
        list(range(0, 20)),
        list(range(20, 40)),
        list(range(40, 60)),
        list(range(60, 80)),
        list(range(80, 101))
    ]

    if args.max_stages is not None:

        stages = all_stages[
            :args.max_stages
        ]

    else:
        stages = all_stages

    print("\nStages:")

    for i, classes in enumerate(
        stages,
        start=1
    ):

        print(
            f"S{i}: classes "
            f"{classes[0] + 1}-"
            f"{classes[-1] + 1}"
        )

    # --------------------------------------------------------
    # Global model
    # --------------------------------------------------------

    print(
        "\nLoading global "
        "K400-pretrained TimeSformer..."
    )

    global_model = (
        TimesformerForVideoClassification
        .from_pretrained(
            (
                "facebook/"
                "timesformer-base-"
                "finetuned-k400"
            ),
            num_labels=101,
            ignore_mismatched_sizes=True
        )
    )

    start_layer, total_layers = (
        configure_partial_finetuning(
            global_model,
            args.last_blocks
        )
    )

    global_model.to(device)

    total_params = sum(
        p.numel()
        for p in global_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in global_model.parameters()
        if p.requires_grad
    )

    print(
        f"Transformer layers : {total_layers}"
    )

    print(
        f"Trainable layers   : "
        f"{start_layer}-{total_layers - 1}"
    )

    print(
        f"Total parameters   : "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"Trainable params    : "
        f"{trainable_params / 1e6:.3f} M"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    os.makedirs(
        args.results_dir,
        exist_ok=True
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    num_stages = len(stages)

    accuracy_matrix = np.full(
        (
            num_stages,
            num_stages
        ),
        np.nan
    )

    f1_matrix = np.full(
        (
            num_stages,
            num_stages
        ),
        np.nan
    )

    round_history = []

    # ========================================================
    # Continual stages
    # ========================================================

    for stage_idx, current_classes in enumerate(
        stages
    ):

        stage_number = stage_idx + 1

        print("\n" + "=" * 76)

        print(
            f"FEDERATED STAGE "
            f"{stage_number}/{num_stages}"
        )

        print("=" * 76)

        stage_dataset = UCFStageDataset(
            args.data_root,
            train_file,
            class_file,
            current_classes,
            frames=args.frames,
            size=args.image_size,
            train=True
        )

        print(
            "Stage training samples :",
            len(stage_dataset)
        )

        # ----------------------------------------------------
        # Fixed non-IID client partition for this stage
        # ----------------------------------------------------

        client_indices = dirichlet_partition(
            stage_dataset,
            args.clients,
            args.alpha,
            args.seed + stage_idx
        )

        print("\nClient partitions:")

        for cid, indices in enumerate(
            client_indices
        ):

            print(
                f"  Client {cid + 1}: "
                f"{len(indices)} samples"
            )

        # ====================================================
        # Communication rounds
        # ====================================================

        for round_idx in range(
            args.rounds_per_stage
        ):

            print(
                f"\n--- Stage {stage_number} | "
                f"Round {round_idx + 1}/"
                f"{args.rounds_per_stage} ---"
            )

            # CPU snapshot avoids keeping
            # multiple 121M models on GPU.
            global_state_cpu = OrderedDict(
                (
                    key,
                    value.detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in global_model.state_dict().items()
            )

            client_states = []
            client_weights = []

            for client_id in range(
                args.clients
            ):

                indices = client_indices[
                    client_id
                ]

                if len(indices) == 0:

                    print(
                        f"Client {client_id + 1}: "
                        f"empty -> skipped"
                    )

                    continue

                print(
                    f"\nClient "
                    f"{client_id + 1}/"
                    f"{args.clients} "
                    f"| samples={len(indices)}"
                )

                # --------------------------------------------
                # One local model at a time
                # --------------------------------------------

                local_model = (
                    TimesformerForVideoClassification
                    .from_pretrained(
                        (
                            "facebook/"
                            "timesformer-base-"
                            "finetuned-k400"
                        ),
                        num_labels=101,
                        ignore_mismatched_sizes=True
                    )
                )

                configure_partial_finetuning(
                    local_model,
                    args.last_blocks
                )

                local_model.load_state_dict(
                    global_state_cpu,
                    strict=True
                )

                local_model.to(device)

                client_subset = Subset(
                    stage_dataset,
                    indices
                )

                client_loader = DataLoader(
                    client_subset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=(
                        args.workers > 0
                    )
                )

                local_loss = train_local_client(
                    model=local_model,
                    loader=client_loader,
                    device=device,
                    local_epochs=args.local_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    amp=args.amp,
                    max_batches=(
                        args.max_local_batches
                    )
                )

                local_state = (
                    get_trainable_state(
                        local_model
                    )
                )

                # In smoke mode weight by
                # actually processed examples.
                if (
                    args.max_local_batches
                    is not None
                ):

                    effective_samples = min(
                        len(indices),
                        args.max_local_batches
                        * args.batch_size
                    )

                else:
                    effective_samples = len(
                        indices
                    )

                client_states.append(
                    local_state
                )

                client_weights.append(
                    effective_samples
                )

                print(
                    f"Client {client_id + 1} "
                    f"loss: {local_loss:.4f}"
                )

                round_history.append(
                    [
                        stage_number,
                        round_idx + 1,
                        client_id + 1,
                        len(indices),
                        effective_samples,
                        local_loss
                    ]
                )

                del local_model
                del client_loader
                del client_subset

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # ------------------------------------------------
            # Weighted FedAvg
            # ------------------------------------------------

            if len(client_states) == 0:
                raise RuntimeError(
                    "No client updates available."
                )

            averaged_state = fedavg(
                client_states,
                client_weights
            )

            load_trainable_state(
                global_model,
                averaged_state
            )

            print(
                "\nFedAvg aggregation complete."
            )

            del client_states
            del averaged_state
            del global_state_cpu

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Save stage checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"e3a_stage{stage_number}.pt"
        )

        torch.save(
            {
                "stage": stage_number,
                "model": global_model.state_dict(),
                "clients": args.clients,
                "alpha": args.alpha,
                "rounds_per_stage": (
                    args.rounds_per_stage
                ),
                "local_epochs": (
                    args.local_epochs
                ),
                "args": vars(args)
            },
            checkpoint_path
        )

        print(
            "\nSaved checkpoint:",
            checkpoint_path
        )

        # ----------------------------------------------------
        # Evaluate all seen stages
        # ----------------------------------------------------

        seen_classes = []

        for j in range(
            stage_idx + 1
        ):
            seen_classes.extend(
                stages[j]
            )

        print(
            f"\nEvaluating "
            f"{stage_idx + 1} "
            f"seen stage(s)..."
        )

        for eval_idx in range(
            stage_idx + 1
        ):

            eval_classes = stages[
                eval_idx
            ]

            test_dataset = UCFStageDataset(
                args.data_root,
                test_file,
                class_file,
                eval_classes,
                frames=args.frames,
                size=args.image_size,
                train=False
            )

            # Smoke evaluation can otherwise
            # take longer than the actual smoke train.
            if (
                args.smoke_test
                and args.smoke_eval_samples
                is not None
            ):

                n_eval = min(
                    args.smoke_eval_samples,
                    len(test_dataset)
                )

                test_dataset = Subset(
                    test_dataset,
                    list(range(n_eval))
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

            acc, macro_f1 = evaluate(
                global_model,
                test_loader,
                device,
                seen_classes,
                eval_classes,
                args.amp
            )

            accuracy_matrix[
                stage_idx,
                eval_idx
            ] = acc

            f1_matrix[
                stage_idx,
                eval_idx
            ] = macro_f1

            print(
                f"After S{stage_number} "
                f"-> Test S{eval_idx + 1}: "
                f"Acc={acc:.3f}% "
                f"F1={macro_f1:.3f}%"
            )

    # ========================================================
    # Final metrics
    # ========================================================

    final_avg_accuracy = float(
        np.nanmean(
            accuracy_matrix[-1, :]
        )
    )

    final_avg_f1 = float(
        np.nanmean(
            f1_matrix[-1, :]
        )
    )

    forgetting_values = []
    bwt_values = []

    if num_stages > 1:

        for j in range(
            num_stages - 1
        ):

            historical = accuracy_matrix[
                j:num_stages - 1,
                j
            ]

            historical = historical[
                ~np.isnan(historical)
            ]

            best_previous = float(
                np.max(historical)
            )

            final_value = float(
                accuracy_matrix[-1, j]
            )

            forgetting_values.append(
                best_previous
                - final_value
            )

            bwt_values.append(
                float(
                    accuracy_matrix[-1, j]
                    - accuracy_matrix[j, j]
                )
            )

        avg_forgetting = float(
            np.mean(
                forgetting_values
            )
        )

        bwt = float(
            np.mean(
                bwt_values
            )
        )

    else:

        avg_forgetting = float("nan")
        bwt = float("nan")

    print("\n" + "=" * 76)

    print(
        "E3-A FINAL RESULTS"
    )

    print("=" * 76)

    print(
        f"Final Average Accuracy : "
        f"{final_avg_accuracy:.3f}%"
    )

    print(
        f"Final Average Macro-F1 : "
        f"{final_avg_f1:.3f}%"
    )

    if num_stages > 1:

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
            accuracy_matrix,
            precision=3,
            suppress_small=True
        )
    )

    print("\nMacro-F1 matrix:")

    print(
        np.array2string(
            f1_matrix,
            precision=3,
            suppress_small=True
        )
    )

    # ========================================================
    # Save
    # ========================================================

    accuracy_file = os.path.join(
        args.results_dir,
        "e3a_accuracy_matrix.csv"
    )

    f1_file = os.path.join(
        args.results_dir,
        "e3a_f1_matrix.csv"
    )

    summary_file = os.path.join(
        args.results_dir,
        "e3a_summary.csv"
    )

    client_file = os.path.join(
        args.results_dir,
        "e3a_client_history.csv"
    )

    np.savetxt(
        accuracy_file,
        accuracy_matrix,
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        f1_file,
        f1_matrix,
        delimiter=",",
        fmt="%.6f"
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "clients",
                "alpha",
                "stages",
                "rounds_per_stage",
                "local_epochs",
                "final_avg_accuracy",
                "final_avg_macro_f1",
                "avg_forgetting_pp",
                "bwt_pp"
            ]
        )

        writer.writerow(
            [
                args.clients,
                args.alpha,
                num_stages,
                args.rounds_per_stage,
                args.local_epochs,
                final_avg_accuracy,
                final_avg_f1,
                avg_forgetting,
                bwt
            ]
        )

    with open(
        client_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "stage",
                "round",
                "client",
                "partition_samples",
                "effective_samples",
                "local_loss"
            ]
        )

        writer.writerows(
            round_history
        )

    print("\nSaved:")
    print(" ", accuracy_file)
    print(" ", f1_file)
    print(" ", summary_file)
    print(" ", client_file)

    if args.smoke_test:

        print(
            "\n"
            + "=" * 76
        )

        print(
            "SMOKE TEST COMPLETED"
        )

        print(
            "Federated partitioning, "
            "local training, FedAvg and "
            "evaluation are working."
        )

        print("=" * 76)


# ============================================================
# Arguments
# ============================================================

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
        "--clients",
        type=int,
        default=5
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--rounds-per-stage",
        type=int,
        default=1
    )

    parser.add_argument(
        "--local-epochs",
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
        "--max-stages",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max-local-batches",
        type=int,
        default=None
    )

    parser.add_argument(
        "--smoke-eval-samples",
        type=int,
        default=40
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/e3a"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results"
    )

    args = parser.parse_args()

    main(args)