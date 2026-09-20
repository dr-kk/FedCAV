import os
import csv
import copy
import random
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import TimesformerForVideoClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# UCF101 Stage Dataset
# =========================================================
class UCFStageDataset(Dataset):

    def __init__(
        self,
        root,
        split_file,
        class_file,
        allowed_classes,
        frames=8,
        size=224,
        train=False,
    ):

        self.root = root
        self.frames = frames
        self.size = size
        self.train = train

        self.allowed_classes = set(allowed_classes)

        self.class_to_idx = {}

        with open(
            class_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                idx, name = line.strip().split()

                self.class_to_idx[name] = (
                    int(idx) - 1
                )

        self.samples = []

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                rel_path = parts[0]

                if len(parts) > 1:

                    label = (
                        int(parts[1]) - 1
                    )

                else:

                    class_name = (
                        rel_path
                        .replace("\\", "/")
                        .split("/")[0]
                    )

                    label = (
                        self.class_to_idx[
                            class_name
                        ]
                    )

                if label not in self.allowed_classes:
                    continue

                full_path = os.path.join(
                    root,
                    rel_path.replace(
                        "/",
                        os.sep
                    )
                )

                self.samples.append(
                    (
                        full_path,
                        label
                    )
                )

    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------
    # Read uniformly sampled frames
    # -----------------------------------------------------
    def _read_video(self, path):

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():

            raise RuntimeError(
                f"Cannot open video: {path}"
            )

        length = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
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

                    frame = (
                        frames[-1].copy()
                    )

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

    # -----------------------------------------------------
    # Transform
    # -----------------------------------------------------
    def _transform(self, frames):

        mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ]
        ).view(3, 1, 1)

        std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ]
        ).view(3, 1, 1)

        # Same horizontal flip for whole clip
        do_flip = (
            self.train
            and random.random() < 0.5
        )

        output = []

        for frame in frames:

            x = torch.from_numpy(
                frame
            )

            x = (
                x.permute(2, 0, 1)
                .float()
                / 255.0
            )

            x = F.interpolate(
                x.unsqueeze(0),
                size=(
                    self.size,
                    self.size
                ),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            if do_flip:

                x = torch.flip(
                    x,
                    dims=[2]
                )

            x = (
                (x - mean)
                / std
            )

            output.append(x)

        return torch.stack(
            output,
            dim=0
        )

    def __getitem__(self, index):

        path, label = (
            self.samples[index]
        )

        frames = self._read_video(
            path
        )

        pixel_values = (
            self._transform(frames)
        )

        return (
            pixel_values,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# =========================================================
# Partial Fine-Tuning
# =========================================================
def configure_partial_finetuning(
    model,
    last_blocks=2
):

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Classifier trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    layers = (
        model.timesformer.encoder.layer
    )

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

    return (
        start_idx,
        total_layers
    )


# =========================================================
# Extract CLS Features
# =========================================================
@torch.no_grad()
def extract_features(
    model,
    loader,
    device,
    amp
):

    model.eval()

    feature_list = []
    label_list = []

    for videos, labels in tqdm(
        loader,
        desc="Extract prototypes",
        leave=False
    ):

        videos = videos.to(
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

            outputs = (
                model.timesformer(
                    pixel_values=videos
                )
            )

            # TimeSformer CLS token
            features = (
                outputs
                .last_hidden_state[:, 0, :]
            )

        # Store prototypes in FP32
        features = (
            features
            .float()
        )

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

        feature_list.append(
            features.cpu()
        )

        label_list.append(
            labels.cpu()
        )

    return (
        torch.cat(
            feature_list,
            dim=0
        ),
        torch.cat(
            label_list,
            dim=0
        )
    )


# =========================================================
# Prototype Selection
# =========================================================
def select_prototypes(
    features,
    labels,
    prototypes_per_class
):

    memory_features = []
    memory_labels = []

    unique_classes = (
        torch.unique(labels)
    )

    for class_id in unique_classes:

        indices = torch.where(
            labels == class_id
        )[0]

        class_features = (
            features[indices]
        )

        # Normalized class centroid
        centroid = (
            class_features
            .mean(
                dim=0,
                keepdim=True
            )
        )

        centroid = F.normalize(
            centroid,
            p=2,
            dim=1
        )

        # Cosine similarity because
        # all representations normalized
        similarity = torch.mm(
            class_features,
            centroid.t()
        ).squeeze(1)

        k = min(
            prototypes_per_class,
            len(indices)
        )

        selected_local = (
            torch.topk(
                similarity,
                k=k,
                largest=True
            ).indices
        )

        selected_features = (
            class_features[
                selected_local
            ]
        )

        selected_labels = (
            torch.full(
                (k,),
                int(class_id.item()),
                dtype=torch.long
            )
        )

        memory_features.append(
            selected_features
        )

        memory_labels.append(
            selected_labels
        )

    return (
        torch.cat(
            memory_features,
            dim=0
        ),
        torch.cat(
            memory_labels,
            dim=0
        )
    )


# =========================================================
# Train One Continual Stage
# =========================================================
def train_stage(
    model,
    loader,
    device,
    epochs,
    lr,
    weight_decay,
    amp,
    proto_features,
    proto_labels,
    teacher_classifier,
    proto_batch,
    lambda_proto,
    lambda_stability,
    temperature
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

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        running_loss = 0.0
        running_current = 0.0
        running_proto = 0.0
        running_stability = 0.0

        seen = 0

        bar = tqdm(
            loader,
            desc=(
                f"Train epoch "
                f"{epoch}/{epochs}"
            )
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
                    amp
                    and device.type == "cuda"
                )
            ):

                logits = model(
                    pixel_values=videos
                ).logits

                current_loss = (
                    criterion(
                        logits,
                        labels
                    )
                )

                proto_loss = torch.tensor(
                    0.0,
                    device=device
                )

                stability_loss = (
                    torch.tensor(
                        0.0,
                        device=device
                    )
                )

                # -----------------------------------------
                # Prototype rehearsal
                # -----------------------------------------
                if (
                    proto_features is not None
                    and proto_labels is not None
                    and len(proto_features) > 0
                ):

                    n_proto = min(
                        proto_batch,
                        len(proto_features)
                    )

                    sampled_indices = (
                        torch.randint(
                            low=0,
                            high=len(
                                proto_features
                            ),
                            size=(n_proto,)
                        )
                    )

                    sampled_features = (
                        proto_features[
                            sampled_indices
                        ]
                        .to(
                            device,
                            non_blocking=True
                        )
                    )

                    sampled_labels = (
                        proto_labels[
                            sampled_indices
                        ]
                        .to(
                            device,
                            non_blocking=True
                        )
                    )

                    # Stored prototypes are
                    # normalized 768-D CLS features.
                    proto_logits = (
                        model.classifier(
                            sampled_features
                        )
                    )

                    proto_loss = criterion(
                        proto_logits,
                        sampled_labels
                    )

                    # -------------------------------------
                    # Stability / distillation
                    # -------------------------------------
                    if (
                        teacher_classifier
                        is not None
                    ):

                        with torch.no_grad():

                            teacher_logits = (
                                teacher_classifier(
                                    sampled_features
                                )
                            )

                        student_log_probs = (
                            F.log_softmax(
                                proto_logits
                                / temperature,
                                dim=1
                            )
                        )

                        teacher_probs = (
                            F.softmax(
                                teacher_logits
                                / temperature,
                                dim=1
                            )
                        )

                        stability_loss = (
                            F.kl_div(
                                student_log_probs,
                                teacher_probs,
                                reduction=(
                                    "batchmean"
                                )
                            )
                            * (
                                temperature
                                ** 2
                            )
                        )

                loss = (
                    current_loss
                    + lambda_proto
                    * proto_loss
                    + lambda_stability
                    * stability_loss
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_size = (
                labels.size(0)
            )

            seen += batch_size

            running_loss += (
                loss.item()
                * batch_size
            )

            running_current += (
                current_loss.item()
                * batch_size
            )

            running_proto += (
                proto_loss.item()
                * batch_size
            )

            running_stability += (
                stability_loss.item()
                * batch_size
            )

            bar.set_postfix(
                loss=(
                    running_loss
                    / seen
                ),
                cls=(
                    running_current
                    / seen
                ),
                proto=(
                    running_proto
                    / seen
                )
            )


# =========================================================
# Evaluation
# =========================================================
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

    # Future unseen classes masked.
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

        # Class-incremental protocol:
        # prediction allowed only among
        # classes observed so far.
        logits[
            :,
            unseen_mask
        ] = -1e9

        predictions = (
            logits.argmax(
                dim=1
            )
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

    # IMPORTANT:
    # Macro-F1 is calculated only
    # across classes belonging to
    # this evaluated stage.
    macro_f1 = (
        f1_score(
            truth,
            predictions_all,
            labels=sorted(
                eval_classes
            ),
            average="macro",
            zero_division=0
        )
        * 100.0
    )

    return (
        accuracy,
        macro_f1
    )


# =========================================================
# Prototype Memory Statistics
# =========================================================
def memory_statistics(
    proto_features,
    proto_labels
):

    if (
        proto_features is None
        or len(proto_features) == 0
    ):

        return (
            0,
            0,
            0.0,
            0.0
        )

    vectors = len(
        proto_features
    )

    classes = len(
        torch.unique(
            proto_labels
        )
    )

    bytes_used = (
        proto_features.numel()
        * proto_features.element_size()
        + proto_labels.numel()
        * proto_labels.element_size()
    )

    mib = (
        bytes_used
        / (1024 ** 2)
    )

    mb = (
        bytes_used
        / 1_000_000
    )

    return (
        vectors,
        classes,
        mib,
        mb
    )


# =========================================================
# Main
# =========================================================
def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(
        "FedCAV E2-B — "
        "Continual Video Learning "
        "with Prototype Memory"
    )
    print("=" * 72)

    print(
        "Device :",
        device
    )

    if device.type == "cuda":

        print(
            "GPU    :",
            torch.cuda.get_device_name(
                0
            )
        )

    # -----------------------------------------------------
    # Files
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Deterministic 5-stage protocol
    # -----------------------------------------------------
    stages = [
        list(range(0, 20)),
        list(range(20, 40)),
        list(range(40, 60)),
        list(range(60, 80)),
        list(range(80, 101))
    ]

    print("\nContinual stages:")

    for i, classes in enumerate(
        stages,
        start=1
    ):

        print(
            f"S{i}: classes "
            f"{classes[0] + 1}-"
            f"{classes[-1] + 1} "
            f"({len(classes)} classes)"
        )

    print(
        "\nPrototype configuration:"
    )

    print(
        "Prototypes/class  :",
        args.prototypes_per_class
    )

    print(
        "Prototype batch   :",
        args.proto_batch
    )

    print(
        "Lambda prototype  :",
        args.lambda_proto
    )

    print(
        "Lambda stability  :",
        args.lambda_stability
    )

    print(
        "Temperature       :",
        args.temperature
    )

    # -----------------------------------------------------
    # Fresh K400 initialization
    # -----------------------------------------------------
    print(
        "\nLoading K400-pretrained "
        "TimeSformer..."
    )

    model = (
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
            model,
            last_blocks=args.last_blocks
        )
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
        f"Transformer layers : "
        f"{total_layers}"
    )

    print(
        f"Trainable layers   : "
        f"{start_layer}-"
        f"{total_layers - 1}"
    )

    print(
        f"Total parameters   : "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"Trainable params    : "
        f"{trainable_params / 1e6:.3f} M"
    )

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------
    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        args.results_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Continual matrices
    # -----------------------------------------------------
    num_stages = len(stages)

    accuracy_matrix = np.full(
        (
            num_stages,
            num_stages
        ),
        np.nan,
        dtype=np.float64
    )

    f1_matrix = np.full(
        (
            num_stages,
            num_stages
        ),
        np.nan,
        dtype=np.float64
    )

    # -----------------------------------------------------
    # Prototype memory is CPU resident
    # -----------------------------------------------------
    proto_features = None
    proto_labels = None

    memory_history = []

    # =====================================================
    # Sequential stages
    # =====================================================
    for stage_idx, current_classes in enumerate(
        stages
    ):

        stage_number = (
            stage_idx + 1
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"TRAINING STAGE "
            f"{stage_number}/"
            f"{num_stages}"
        )

        print(
            "=" * 72
        )

        # -------------------------------------------------
        # Teacher = classifier state BEFORE new stage
        # -------------------------------------------------
        teacher_classifier = None

        if (
            proto_features is not None
            and len(proto_features) > 0
        ):

            teacher_classifier = (
                copy.deepcopy(
                    model.classifier
                )
            )

            teacher_classifier = (
                teacher_classifier.to(
                    device
                )
            )

            teacher_classifier.eval()

            for p in (
                teacher_classifier.parameters()
            ):

                p.requires_grad = False

        # -------------------------------------------------
        # Current-stage training data
        # -------------------------------------------------
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
            "Training samples :",
            len(train_dataset)
        )

        vectors, memory_classes, mib, mb = (
            memory_statistics(
                proto_features,
                proto_labels
            )
        )

        print(
            "Memory before stage:"
        )

        print(
            f"  prototypes : "
            f"{vectors}"
        )

        print(
            f"  classes    : "
            f"{memory_classes}"
        )

        print(
            f"  size       : "
            f"{mib:.3f} MiB "
            f"({mb:.3f} MB)"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------
        train_stage(
            model=model,
            loader=train_loader,
            device=device,
            epochs=(
                args.epochs_per_stage
            ),
            lr=args.lr,
            weight_decay=(
                args.weight_decay
            ),
            amp=args.amp,
            proto_features=(
                proto_features
            ),
            proto_labels=(
                proto_labels
            ),
            teacher_classifier=(
                teacher_classifier
            ),
            proto_batch=(
                args.proto_batch
            ),
            lambda_proto=(
                args.lambda_proto
            ),
            lambda_stability=(
                args.lambda_stability
            ),
            temperature=(
                args.temperature
            )
        )

        # Free teacher GPU memory
        if teacher_classifier is not None:

            del teacher_classifier

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # -------------------------------------------------
        # Build prototypes from current stage
        # using deterministic non-augmented clips.
        # -------------------------------------------------
        prototype_dataset = (
            UCFStageDataset(
                args.data_root,
                train_file,
                class_file,
                current_classes,
                frames=args.frames,
                size=args.image_size,
                train=False
            )
        )

        prototype_loader = DataLoader(
            prototype_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(
                args.workers > 0
            )
        )

        print(
            "\nExtracting current-stage "
            "prototype candidates..."
        )

        current_features, current_labels = (
            extract_features(
                model,
                prototype_loader,
                device,
                args.amp
            )
        )

        new_proto_features, new_proto_labels = (
            select_prototypes(
                current_features,
                current_labels,
                args.prototypes_per_class
            )
        )

        # -------------------------------------------------
        # Add current-stage prototypes
        # -------------------------------------------------
        if proto_features is None:

            proto_features = (
                new_proto_features
                .cpu()
                .float()
            )

            proto_labels = (
                new_proto_labels
                .cpu()
                .long()
            )

        else:

            proto_features = (
                torch.cat(
                    [
                        proto_features,
                        new_proto_features
                        .cpu()
                        .float()
                    ],
                    dim=0
                )
            )

            proto_labels = (
                torch.cat(
                    [
                        proto_labels,
                        new_proto_labels
                        .cpu()
                        .long()
                    ],
                    dim=0
                )
            )

        vectors, memory_classes, mib, mb = (
            memory_statistics(
                proto_features,
                proto_labels
            )
        )

        print(
            "\nPrototype memory after "
            f"S{stage_number}:"
        )

        print(
            f"  vectors : {vectors}"
        )

        print(
            f"  classes : {memory_classes}"
        )

        print(
            f"  size    : "
            f"{mib:.3f} MiB "
            f"({mb:.3f} MB)"
        )

        memory_history.append(
            [
                stage_number,
                vectors,
                memory_classes,
                mib,
                mb
            ]
        )

        # -------------------------------------------------
        # Save stage checkpoint + memory
        # -------------------------------------------------
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            (
                f"e2b_stage"
                f"{stage_number}.pt"
            )
        )

        torch.save(
            {
                "stage": stage_number,
                "model": (
                    model.state_dict()
                ),
                "prototype_features": (
                    proto_features
                ),
                "prototype_labels": (
                    proto_labels
                ),
                "memory_mib": mib,
                "memory_mb": mb,
                "args": vars(args)
            },
            checkpoint_path
        )

        print(
            "Saved checkpoint:",
            checkpoint_path
        )

        # -------------------------------------------------
        # Seen classes through current stage
        # -------------------------------------------------
        seen_classes = []

        for j in range(
            stage_idx + 1
        ):

            seen_classes.extend(
                stages[j]
            )

        # -------------------------------------------------
        # Evaluate every previously learned stage
        # -------------------------------------------------
        print(
            f"\nEvaluating "
            f"{stage_idx + 1} "
            f"seen stage(s)..."
        )

        for eval_idx in range(
            stage_idx + 1
        ):

            eval_classes = (
                stages[eval_idx]
            )

            test_dataset = (
                UCFStageDataset(
                    args.data_root,
                    test_file,
                    class_file,
                    eval_classes,
                    frames=args.frames,
                    size=args.image_size,
                    train=False
                )
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
                model=model,
                loader=test_loader,
                device=device,
                seen_classes=(
                    seen_classes
                ),
                eval_classes=(
                    eval_classes
                ),
                amp=args.amp
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
                f"-> Test S"
                f"{eval_idx + 1}: "
                f"Acc={acc:.3f}% "
                f"F1={macro_f1:.3f}%"
            )

    # =====================================================
    # Final Metrics
    # =====================================================
    final_average_accuracy = float(
        np.nanmean(
            accuracy_matrix[-1, :]
        )
    )

    final_average_f1 = float(
        np.nanmean(
            f1_matrix[-1, :]
        )
    )

    # -----------------------------------------------------
    # Forgetting
    # F_j = max_{l < T} A[l,j] - A[T,j]
    # Only old tasks included.
    # -----------------------------------------------------
    forgetting_values = []

    for j in range(
        num_stages - 1
    ):

        historical_values = (
            accuracy_matrix[
                j:num_stages - 1,
                j
            ]
        )

        historical_values = (
            historical_values[
                ~np.isnan(
                    historical_values
                )
            ]
        )

        best_previous = float(
            np.max(
                historical_values
            )
        )

        final_value = float(
            accuracy_matrix[
                -1,
                j
            ]
        )

        forgetting_values.append(
            best_previous
            - final_value
        )

    average_forgetting = float(
        np.mean(
            forgetting_values
        )
    )

    # -----------------------------------------------------
    # Backward Transfer
    # BWT = mean_j A[T,j] - A[j,j]
    # -----------------------------------------------------
    bwt_values = []

    for j in range(
        num_stages - 1
    ):

        bwt_values.append(
            float(
                accuracy_matrix[
                    -1,
                    j
                ]
                - accuracy_matrix[
                    j,
                    j
                ]
            )
        )

    bwt = float(
        np.mean(
            bwt_values
        )
    )

    # -----------------------------------------------------
    # Final memory
    # -----------------------------------------------------
    final_vectors, final_classes, final_mib, final_mb = (
        memory_statistics(
            proto_features,
            proto_labels
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "E2-B FINAL RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"Final Average Accuracy : "
        f"{final_average_accuracy:.3f}%"
    )

    print(
        f"Final Average Macro-F1 : "
        f"{final_average_f1:.3f}%"
    )

    print(
        f"Average Forgetting     : "
        f"{average_forgetting:.3f} pp"
    )

    print(
        f"Backward Transfer      : "
        f"{bwt:.3f} pp"
    )

    print(
        f"Final prototypes       : "
        f"{final_vectors}"
    )

    print(
        f"Prototype classes      : "
        f"{final_classes}"
    )

    print(
        f"Prototype memory       : "
        f"{final_mib:.3f} MiB "
        f"({final_mb:.3f} MB)"
    )

    print(
        "\nAccuracy matrix:"
    )

    print(
        np.array2string(
            accuracy_matrix,
            precision=3,
            suppress_small=True
        )
    )

    print(
        "\nMacro-F1 matrix:"
    )

    print(
        np.array2string(
            f1_matrix,
            precision=3,
            suppress_small=True
        )
    )

    # =====================================================
    # Save Results
    # =====================================================
    accuracy_file = os.path.join(
        args.results_dir,
        "e2b_accuracy_matrix.csv"
    )

    f1_file = os.path.join(
        args.results_dir,
        "e2b_f1_matrix.csv"
    )

    summary_file = os.path.join(
        args.results_dir,
        "e2b_summary.csv"
    )

    memory_file = os.path.join(
        args.results_dir,
        "e2b_memory_history.csv"
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
                "final_avg_accuracy",
                "final_avg_macro_f1",
                "avg_forgetting_pp",
                "bwt_pp",
                "prototypes_per_class",
                "final_prototypes",
                "memory_mib",
                "memory_mb"
            ]
        )

        writer.writerow(
            [
                final_average_accuracy,
                final_average_f1,
                average_forgetting,
                bwt,
                args.prototypes_per_class,
                final_vectors,
                final_mib,
                final_mb
            ]
        )

    with open(
        memory_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "stage",
                "prototype_vectors",
                "represented_classes",
                "memory_mib",
                "memory_mb"
            ]
        )

        writer.writerows(
            memory_history
        )

    print(
        "\nSaved:"
    )

    print(
        " ",
        accuracy_file
    )

    print(
        " ",
        f1_file
    )

    print(
        " ",
        summary_file
    )

    print(
        " ",
        memory_file
    )


# =========================================================
# Arguments
# =========================================================
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

    # -----------------------------------------------------
    # Prototype parameters
    # -----------------------------------------------------
    parser.add_argument(
        "--prototypes-per-class",
        type=int,
        default=10
    )

    parser.add_argument(
        "--proto-batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lambda-proto",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--lambda-stability",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/e2b"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results"
    )

    args = parser.parse_args()

    main(args)