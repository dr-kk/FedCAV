import os
import csv
import random
import argparse
from collections import OrderedDict, defaultdict

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# UCF101 Dataset
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
                        rel_path.replace("\\", "/").split("/")[0]
                    )
                    label = self.class_to_idx[class_name]

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

        frames = self._read_video(
            path
        )

        pixel_values = self._transform(
            frames
        )

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

    return (
        start_idx,
        total_layers
    )


# ============================================================
# Dirichlet Non-IID Partition
# ============================================================

def dirichlet_partition(
    dataset,
    num_clients,
    alpha,
    seed
):

    rng = np.random.default_rng(
        seed
    )

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

    unique_classes = np.unique(
        labels
    )

    for class_id in unique_classes:

        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

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

        for client_id, count in enumerate(
            counts
        ):

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
# Prototype Bank Statistics
# ============================================================

def prototype_bank_size(
    prototype_bank
):

    vectors = sum(
        len(v)
        for v in prototype_bank.values()
    )

    classes = len(
        prototype_bank
    )

    bytes_used = 0

    for vector_list in prototype_bank.values():

        for vector in vector_list:

            bytes_used += (
                vector.numel()
                * vector.element_size()
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


# ============================================================
# Prototype Replay
# ============================================================

def sample_prototype_replay(
    prototype_bank,
    batch_size,
    device
):

    if len(prototype_bank) == 0:
        return None, None

    features = []
    labels = []

    for class_id, vectors in (
        prototype_bank.items()
    ):

        for vector in vectors:

            features.append(
                vector
            )

            labels.append(
                class_id
            )

    if len(features) == 0:
        return None, None

    n = min(
        batch_size,
        len(features)
    )

    selected = random.sample(
        range(len(features)),
        n
    )

    replay_features = torch.stack(
        [
            features[i]
            for i in selected
        ],
        dim=0
    ).float().to(device)

    replay_labels = torch.tensor(
        [
            labels[i]
            for i in selected
        ],
        dtype=torch.long,
        device=device
    )

    return (
        replay_features,
        replay_labels
    )


# ============================================================
# Proximal Stabilization
# ============================================================

def proximal_regularization(
    model,
    global_reference,
    device
):

    prox_loss = torch.zeros(
        (),
        device=device
    )

    for name, param in (
        model.named_parameters()
    ):

        if not param.requires_grad:
            continue

        reference = (
            global_reference[name]
            .to(
                device=device,
                dtype=param.dtype
            )
        )

        prox_loss = prox_loss + torch.sum(
            (
                param
                - reference
            ) ** 2
        )

    return (
        0.5
        * prox_loss
    )


# ============================================================
# E4 Drift-Adaptive Stabilization
# ============================================================

def parameter_drift(local_state, global_reference):
    """Relative L2 drift over trainable tensors."""
    num = 0.0
    den = 0.0
    for name, local_tensor in local_state.items():
        ref = global_reference[name].detach().cpu().float()
        cur = local_tensor.detach().cpu().float()
        num += torch.sum((cur - ref) ** 2).item()
        den += torch.sum(ref ** 2).item()
    return float(np.sqrt(num / max(den, 1e-12)))


def adaptive_mu(
    base_mu,
    previous_drift,
    drift_ema,
    gamma,
    mu_min,
    mu_max,
    eps=1e-12
):
    """
    Scale-independent drift-adaptive proximal strength.

    r = previous_drift / EMA(previous drift)
    mu = clip(mu0 * [1 + gamma * (r - 1)], mu_min, mu_max)

    On first participation, previous_drift is None and mu=base_mu.
    """
    if previous_drift is None or drift_ema is None or drift_ema <= eps:
        bounded_drift = 1.0
        value = base_mu
    else:
        bounded_drift = float(previous_drift / (drift_ema + eps))
        value = base_mu * (1.0 + gamma * (bounded_drift - 1.0))

    value = float(np.clip(value, mu_min, mu_max))
    return value, bounded_drift


# ============================================================
# E4-v3 Prototype-Space Functional Drift
# ============================================================

@torch.no_grad()
def prototype_functional_drift(
    local_model,
    global_reference,
    prototype_bank,
    device,
    max_prototypes=256,
    eps=1e-8
):
    """Functional classifier drift evaluated on remembered old prototypes."""
    if prototype_bank is None or len(prototype_bank) == 0:
        return 0.0

    flat_features = []

    # Working FedCAV bank is dict-like: class_id -> stored prototype feature(s).
    if isinstance(prototype_bank, dict):
        stored_groups = prototype_bank.values()
    else:
        # Defensive fallback for list/tuple banks.
        stored_groups = prototype_bank

    for group in stored_groups:
        if group is None:
            continue

        if isinstance(group, (list, tuple)):
            items = group
        else:
            items = [group]

        for feat in items:
            if feat is None:
                continue
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat)
            feat = feat.detach().cpu().float()

            if feat.ndim == 1:
                flat_features.append(feat)
            elif feat.ndim == 2:
                flat_features.extend(list(feat))
            else:
                feat = feat.reshape(-1, feat.shape[-1])
                flat_features.extend(list(feat))

    if len(flat_features) == 0:
        return 0.0

    features = torch.stack(flat_features, dim=0)

    if features.shape[0] > max_prototypes:
        idx = torch.linspace(
            0, features.shape[0] - 1, steps=max_prototypes
        ).long()
        features = features[idx]

    features = features.to(device=device, dtype=torch.float32)

    local_w = local_model.classifier.weight.detach().float()
    local_b = local_model.classifier.bias.detach().float()
    global_w = global_reference["classifier.weight"].detach().to(
        device=device, dtype=torch.float32
    )
    global_b = global_reference["classifier.bias"].detach().to(
        device=device, dtype=torch.float32
    )

    logits_local = F.linear(features, local_w, local_b)
    logits_global = F.linear(features, global_w, global_b)

    log_p = F.log_softmax(logits_global, dim=1)
    log_q = F.log_softmax(logits_local, dim=1)
    p = log_p.exp()
    q = log_q.exp()

    kl_pq = torch.sum(p * (log_p - log_q), dim=1)
    kl_qp = torch.sum(q * (log_q - log_p), dim=1)
    drift = 0.5 * (kl_pq + kl_qp)
    drift = drift.mean()

    if not torch.isfinite(drift):
        return 0.0

    return float(drift.clamp_min(0.0).item())


def stability_plasticity_adaptive_mu(
    base_mu,
    previous_drift,
    previous_plasticity,
    drift_scale,
    mu_min,
    mu_max
):
    """
    Causal E4-v6 stability--plasticity controller.

    stability = 1 - exp(-drift_scale * previous_drift)
    mu = base_mu + stability * (1 - previous_plasticity)
         * (mu_max - base_mu)

    previous_plasticity is measured after the client's preceding
    participation as current_loss / (current_loss + replay_loss + eps).
    Thus large current-task pressure backs off stabilization, while
    large old-knowledge drift with low plasticity demand strengthens it.
    """
    if previous_drift is None or previous_drift <= 0.0:
        stability = 0.0
    else:
        scaled = max(0.0, float(drift_scale) * float(previous_drift))
        stability = float(1.0 - np.exp(-scaled))

    if previous_plasticity is None:
        plasticity = 1.0
    else:
        plasticity = float(np.clip(previous_plasticity, 0.0, 1.0))

    mu = float(
        base_mu
        + stability
        * (1.0 - plasticity)
        * (mu_max - base_mu)
    )
    mu = float(np.clip(mu, mu_min, mu_max))
    return mu, stability, plasticity


def compute_plasticity_signal(current_loss, replay_loss, eps=1e-8):
    """Bounded current-task learning-pressure signal in [0, 1]."""
    current = max(0.0, float(current_loss))
    replay = max(0.0, float(replay_loss))
    denom = current + replay + float(eps)
    if denom <= 0.0:
        return 1.0
    return float(np.clip(current / denom, 0.0, 1.0))




def adaptive_replay_weight(base_lambda, previous_drift, previous_plasticity,
                           drift_scale, lambda_min, lambda_max):
    """Causal bounded replay controller using previous-participation signals."""
    if previous_drift is None or previous_drift <= 0.0:
        stability = 0.0
    else:
        stability = float(1.0 - np.exp(
            -max(0.0, float(drift_scale) * float(previous_drift))
        ))
    plasticity = 1.0 if previous_plasticity is None else float(
        np.clip(previous_plasticity, 0.0, 1.0)
    )
    replay_weight = base_lambda + stability * (1.0 - plasticity) * (
        lambda_max - base_lambda
    )
    return float(np.clip(replay_weight, lambda_min, lambda_max)), stability, plasticity


def e8_replay_controller(mode, base_lambda, previous_drift, previous_plasticity,
                         drift_scale, lambda_min, lambda_max):
    """Controlled E8 ablation; only replay weighting changes between variants.

    drift_only holds plasticity at zero (maximum replay demand).
    plasticity_only holds stability at one (maximum drift demand).
    These are signal-removal interventions, not independent learned controllers.
    """
    if mode == "stage_ramp":
        # The stage-dependent weight is assigned in the main training loop.
        return float(base_lambda), 0.0, 1.0
    if mode == "fixed":
        return float(base_lambda), 0.0, 1.0
    if mode in ("fixed_125", "fixed_135", "fixed_150"):
        fixed_value = {"fixed_125": 1.25, "fixed_135": 1.35,
                       "fixed_150": 1.50}[mode]
        return float(fixed_value), 0.0, 1.0
    if mode == "full":
        return adaptive_replay_weight(
            base_lambda, previous_drift, previous_plasticity,
            drift_scale, lambda_min, lambda_max
        )
    if previous_drift is None or previous_drift <= 0.0:
        stability = 0.0
    else:
        stability = float(1.0 - np.exp(
            -max(0.0, float(drift_scale) * float(previous_drift))
        ))
    plasticity = (1.0 if previous_plasticity is None else
                  float(np.clip(previous_plasticity, 0.0, 1.0)))
    if mode == "drift_only":
        # Missing historical drift still implies no adaptive adjustment.
        delta = stability
    elif mode == "plasticity_only":
        # Preserve original cold-start: do not adapt until drift is observed.
        delta = 0.0 if previous_drift is None else (1.0 - plasticity)
    else:
        raise ValueError(f"Unknown controller mode: {mode}")
    weight = base_lambda + delta * (lambda_max - base_lambda)
    return float(np.clip(weight, lambda_min, lambda_max)), stability, plasticity


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
    prototype_bank,
    lambda_replay,
    replay_batch,
    global_reference,
    mu_prox,
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
    total_current_loss = 0.0
    total_replay_loss = 0.0
    total_prox_loss = 0.0

    replay_steps = 0
    replay_samples = 0

    for epoch in range(
        1,
        local_epochs + 1
    ):

        model.train()

        bar = tqdm(
            loader,
            desc=(
                f"Local epoch "
                f"{epoch}/{local_epochs}"
            ),
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

                # ----------------------------------------
                # Current-stage classification
                # ----------------------------------------

                outputs = model.timesformer(
                    pixel_values=videos
                )

                current_features = (
                    outputs
                    .last_hidden_state[:, 0, :]
                )

                current_logits = (
                    model.classifier(
                        current_features
                    )
                )

                current_loss = criterion(
                    current_logits,
                    labels
                )

                # ----------------------------------------
                # Historical prototype replay
                # ----------------------------------------

                (
                    replay_features,
                    replay_labels
                ) = sample_prototype_replay(
                    prototype_bank,
                    replay_batch,
                    device
                )

                if replay_features is not None:

                    replay_logits = (
                        model.classifier(
                            replay_features
                        )
                    )

                    replay_loss = criterion(
                        replay_logits,
                        replay_labels
                    )

                    replay_steps += 1

                    replay_samples += (
                        replay_labels.size(0)
                    )

                else:

                    replay_loss = torch.zeros(
                        (),
                        device=device
                    )

                # ----------------------------------------
                # FedProx-style stabilization
                # ----------------------------------------

                prox_loss = proximal_regularization(
                    model,
                    global_reference,
                    device
                )

                # ----------------------------------------
                # Combined objective
                # ----------------------------------------

                loss = (
                    current_loss
                    + lambda_replay
                    * replay_loss
                    + mu_prox
                    * prox_loss
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
                loss.item()
                * bs
            )

            total_current_loss += (
                current_loss.item()
                * bs
            )

            total_replay_loss += (
                replay_loss.item()
                * bs
            )

            total_prox_loss += (
                prox_loss.item()
                * bs
            )

            bar.set_postfix(
                loss=(
                    total_loss
                    / max(total_seen, 1)
                ),
                current=(
                    total_current_loss
                    / max(total_seen, 1)
                ),
                replay=(
                    total_replay_loss
                    / max(total_seen, 1)
                ),
                prox=(
                    total_prox_loss
                    / max(total_seen, 1)
                )
            )

    return {
        "loss": (
            total_loss
            / max(total_seen, 1)
        ),
        "current_loss": (
            total_current_loss
            / max(total_seen, 1)
        ),
        "replay_loss": (
            total_replay_loss
            / max(total_seen, 1)
        ),
        "prox_loss": (
            total_prox_loss
            / max(total_seen, 1)
        ),
        "replay_steps": replay_steps,
        "replay_samples": replay_samples
    }


# ============================================================
# Trainable State
# ============================================================

def get_trainable_state(
    model
):

    trainable_names = {
        name
        for name, param
        in model.named_parameters()
        if param.requires_grad
    }

    state = OrderedDict()

    for name, tensor in (
        model.state_dict().items()
    ):

        if name in trainable_names:

            state[name] = (
                tensor
                .detach()
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


def load_trainable_state(
    model,
    averaged_state
):

    current_state = (
        model.state_dict()
    )

    for key, value in (
        averaged_state.items()
    ):

        current_state[key] = (
            value.to(
                dtype=current_state[
                    key
                ].dtype
            )
        )

    model.load_state_dict(
        current_state,
        strict=True
    )


# ============================================================
# Prototype Extraction
# ============================================================

@torch.no_grad()
def extract_client_prototypes(
    model,
    dataset,
    indices,
    current_classes,
    device,
    batch_size,
    workers,
    amp,
    prototypes_per_class,
    max_samples=None
):

    if len(indices) == 0:
        return {}

    working_indices = list(
        indices
    )

    if max_samples is not None:

        working_indices = (
            working_indices[
                :min(
                    max_samples,
                    len(working_indices)
                )
            ]
        )

    subset = Subset(
        dataset,
        working_indices
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        )
    )

    model.eval()

    features_by_class = defaultdict(
        list
    )

    for videos, labels in tqdm(
        loader,
        desc="Prototype extraction",
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

            outputs = model.timesformer(
                pixel_values=videos
            )

            features = (
                outputs
                .last_hidden_state[:, 0, :]
            )

        features = F.normalize(
            features.float(),
            p=2,
            dim=1
        ).cpu()

        for feature, label in zip(
            features,
            labels
        ):

            class_id = int(
                label.item()
            )

            if class_id in current_classes:

                features_by_class[
                    class_id
                ].append(
                    feature
                )

    client_prototypes = {}

    for class_id, vectors in (
        features_by_class.items()
    ):

        vectors = torch.stack(
            vectors,
            dim=0
        ).float()

        centroid = F.normalize(
            vectors.mean(
                dim=0,
                keepdim=True
            ),
            p=2,
            dim=1
        )

        similarities = torch.mm(
            vectors,
            centroid.t()
        ).squeeze(1)

        k = min(
            prototypes_per_class,
            len(vectors)
        )

        selected = torch.topk(
            similarities,
            k=k,
            largest=True
        ).indices

        client_prototypes[
            class_id
        ] = [
            vectors[i]
            .clone()
            .cpu()
            .float()
            for i in selected
        ]

    del loader
    del subset

    return client_prototypes


# ============================================================
# Prototype Server Merge
# ============================================================

def merge_server_prototypes(
    prototype_bank,
    client_prototype_sets,
    prototypes_per_class
):

    incoming = defaultdict(
        list
    )

    for client_bank in (
        client_prototype_sets
    ):

        for class_id, vectors in (
            client_bank.items()
        ):

            incoming[
                class_id
            ].extend(
                vectors
            )

    for class_id, new_vectors in (
        incoming.items()
    ):

        candidates = []

        if class_id in prototype_bank:

            candidates.extend(
                prototype_bank[
                    class_id
                ]
            )

        candidates.extend(
            new_vectors
        )

        if len(candidates) == 0:
            continue

        candidates = torch.stack(
            candidates,
            dim=0
        ).float()

        candidates = F.normalize(
            candidates,
            p=2,
            dim=1
        )

        centroid = F.normalize(
            candidates.mean(
                dim=0,
                keepdim=True
            ),
            p=2,
            dim=1
        )

        similarities = torch.mm(
            candidates,
            centroid.t()
        ).squeeze(1)

        k = min(
            prototypes_per_class,
            len(candidates)
        )

        selected = torch.topk(
            similarities,
            k=k,
            largest=True
        ).indices

        prototype_bank[
            class_id
        ] = [
            candidates[i]
            .clone()
            .cpu()
            .float()
            for i in selected
        ]

    return prototype_bank


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


# ============================================================
# Main
# ============================================================

def main(args):

    if args.controller_mode == "fixed" and not (
        args.replay_lambda_min <= args.lambda_replay <= args.replay_lambda_max
    ):
        raise ValueError("Base replay lambda must lie within replay bounds")
    if args.results_dir == args.checkpoint_dir:
        raise ValueError("Results and checkpoint directories must differ")
    print("E8 controller mode:", args.controller_mode)

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)

    print(
        "FedCAV E4-v7 â€” Adaptive-Replay "
        "Federated Continual Video Learning"
    )

    print("=" * 80)

    print(
        "Device             :",
        device
    )

    if device.type == "cuda":

        print(
            "GPU                :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Clients            :",
        args.clients
    )

    print(
        "Dirichlet alpha    :",
        args.alpha
    )

    print(
        "Rounds/stage       :",
        args.rounds_per_stage
    )

    print(
        "Local epochs       :",
        args.local_epochs
    )

    print(
        "Last blocks        :",
        args.last_blocks
    )

    print(
        "Prototypes/class   :",
        args.prototypes_per_class
    )

    print(
        "Lambda replay      :",
        args.lambda_replay
    )

    print(
        "Replay batch       :",
        args.replay_batch
    )

    print(
        "Mu proximal        :",
        args.mu_prox
    )

    print(
        "Drift gamma        :",
        args.drift_gamma
    )
    print(
        "Drift EMA beta     :",
        args.drift_beta
    )

    print(
        "Drift proto samples:",
        args.drift_proto_samples
    )
    print(
        "Drift scale        :",
        args.drift_scale
    )
    print(
        "Adaptive mu range  :",
        f"[{args.mu_min}, {args.mu_max}]"
    )

    if args.smoke_test:

        print(
            "Mode               : "
            "SMOKE TEST"
        )

    # ========================================================
    # Files
    # ========================================================

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

    # ========================================================
    # Stages
    # ========================================================

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
            f"{classes[-1] + 1} "
            f"({len(classes)} classes)"
        )

    # ========================================================
    # Global Model
    # ========================================================

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

    (
        start_layer,
        total_layers
    ) = configure_partial_finetuning(
        global_model,
        args.last_blocks
    )

    global_model.to(
        device
    )

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

    os.makedirs(
        args.results_dir,
        exist_ok=True
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    num_stages = len(
        stages
    )

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

    prototype_bank = {}

    client_history = []
    memory_history = []
    drift_history = []
    trace_rows = []

    # Persistent client-specific drift state across rounds/stages.
    client_drift_ema = {
        client_id: None
        for client_id in range(args.clients)
    }

    client_last_drift = {
        client_id: None
        for client_id in range(args.clients)
    }

    # Previous-participation plasticity signal for each client.
    client_last_plasticity = {
        client_id: None
        for client_id in range(args.clients)
    }

    # ========================================================
    # Continual Stages
    # ========================================================

    for stage_idx, current_classes in enumerate(
        stages
    ):

        stage_number = (
            stage_idx + 1
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FEDERATED STAGE "
            f"{stage_number}/"
            f"{num_stages}"
        )

        print(
            "=" * 80
        )

        (
            vectors,
            classes,
            mib,
            mb
        ) = prototype_bank_size(
            prototype_bank
        )

        print(
            "Server memory before stage:"
        )

        print(
            f"  prototypes : "
            f"{vectors}"
        )

        print(
            f"  classes    : "
            f"{classes}"
        )

        print(
            f"  size       : "
            f"{mib:.3f} MiB "
            f"({mb:.3f} MB)"
        )

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

        client_indices = (
            dirichlet_partition(
                stage_dataset,
                args.clients,
                args.alpha,
                args.seed + stage_idx
            )
        )

        print(
            "\nClient partitions:"
        )

        for cid, indices in enumerate(
            client_indices
        ):

            print(
                f"  Client "
                f"{cid + 1}: "
                f"{len(indices)} samples"
            )

        # ====================================================
        # Rounds
        # ====================================================

        for round_idx in range(
            args.rounds_per_stage
        ):

            print(
                f"\n--- Stage "
                f"{stage_number} "
                f"| Round "
                f"{round_idx + 1}/"
                f"{args.rounds_per_stage} ---"
            )

            global_state_cpu = OrderedDict(
                (
                    key,
                    value
                    .detach()
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
                    continue

                print(
                    f"\nClient "
                    f"{client_id + 1}/"
                    f"{args.clients} "
                    f"| samples="
                    f"{len(indices)}"
                )

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

                local_model.to(
                    device
                )

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

                previous_drift = client_last_drift[
                    client_id
                ]
                previous_ema = client_drift_ema[
                    client_id
                ]

                previous_plasticity = client_last_plasticity[
                    client_id
                ]

                # E4-v7: mild fixed proximal stabilization + bounded adaptive replay.
                mu_client = float(args.mu_prox)
                (
                    replay_weight,
                    stability_signal,
                    controller_plasticity
                ) = e8_replay_controller(
                    mode=args.controller_mode,
                    base_lambda=args.lambda_replay,
                    previous_drift=previous_drift,
                    previous_plasticity=previous_plasticity,
                    drift_scale=args.drift_scale,
                    lambda_min=args.replay_lambda_min,
                    lambda_max=args.replay_lambda_max
                )

                # Prespecified stage ramp, independent of observed drift/plasticity.
                # Stage 1 has no historical classes; S2..S5: 1.00,1.15,1.30,1.45.
                if args.controller_mode == "stage_ramp":
                    replay_weight = min(1.45, 1.0 + 0.15 * max(0, stage_number - 2))
                    stability_signal = 0.0
                    controller_plasticity = 1.0
                previous_drift_used = previous_drift
                previous_plasticity_used = previous_plasticity
                preclip_weight = float(replay_weight)
                if args.controller_mode in ("full", "drift_only", "plasticity_only"):
                    if args.controller_mode == "full":
                        delta = stability_signal * (1.0 - controller_plasticity)
                    elif args.controller_mode == "drift_only":
                        delta = stability_signal
                    else:
                        delta = 0.0 if previous_drift is None else (1.0 - controller_plasticity)
                    preclip_weight = args.lambda_replay + delta * (args.replay_lambda_max - args.lambda_replay)
                trace_rows.append([
                    stage_number, round_idx + 1, client_id + 1,
                    args.controller_mode,
                    "" if previous_drift_used is None else repr(float(previous_drift_used)),
                    "" if previous_plasticity_used is None else repr(float(previous_plasticity_used)),
                    repr(float(args.drift_scale)), repr(float(stability_signal)),
                    repr(float(controller_plasticity)), repr(float(preclip_weight)),
                    repr(float(replay_weight)),
                    int(abs(preclip_weight - replay_weight) > 1e-12)
                ])

                stats = train_local_client(
                    model=local_model,
                    loader=client_loader,
                    device=device,
                    local_epochs=(
                        args.local_epochs
                    ),
                    lr=args.lr,
                    weight_decay=(
                        args.weight_decay
                    ),
                    amp=args.amp,
                    prototype_bank=(
                        prototype_bank
                    ),
                    lambda_replay=(
                        replay_weight
                    ),
                    replay_batch=(
                        args.replay_batch
                    ),
                    global_reference=(
                        global_state_cpu
                    ),
                    mu_prox=(
                        mu_client
                    ),
                    max_batches=(
                        args.max_local_batches
                    )
                )

                local_state = (
                    get_trainable_state(
                        local_model
                    )
                )

                raw_drift = prototype_functional_drift(
                    local_model=local_model,
                    global_reference=global_state_cpu,
                    prototype_bank=prototype_bank,
                    device=device,
                    max_prototypes=args.drift_proto_samples
                )

                ema_before_update = client_drift_ema[
                    client_id
                ]

                if ema_before_update is None:
                    updated_ema = raw_drift
                else:
                    updated_ema = (
                        args.drift_beta * ema_before_update
                        + (1.0 - args.drift_beta) * raw_drift
                    )

                client_last_drift[
                    client_id
                ] = raw_drift

                client_drift_ema[
                    client_id
                ] = updated_ema

                measured_plasticity = compute_plasticity_signal(
                    stats["current_loss"],
                    stats["replay_loss"]
                )
                client_last_plasticity[client_id] = measured_plasticity

                drift_history.append(
                    [
                        stage_number,
                        round_idx + 1,
                        client_id + 1,
                        raw_drift,
                        0.0 if ema_before_update is None else ema_before_update,
                        updated_ema,
                        stability_signal,
                        controller_plasticity,
                        measured_plasticity,
                        mu_client,
                        replay_weight
                    ]
                )

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

                    effective_samples = (
                        len(indices)
                    )

                client_states.append(
                    local_state
                )

                client_weights.append(
                    effective_samples
                )

                print(
                    f"Client "
                    f"{client_id + 1} "
                    f"| loss="
                    f"{stats['loss']:.4f} "
                    f"| Current="
                    f"{stats['current_loss']:.4f} "
                    f"| Replay="
                    f"{stats['replay_loss']:.4f} "
                    f"| Prox="
                    f"{stats['prox_loss']:.6f} "
                    f"| ProtoDrift="
                    f"{raw_drift:.10e} "
                    f"| ProtoDriftEMA="
                    f"{updated_ema:.10e} "
                    f"| Stability="
                    f"{stability_signal:.4f} "
                    f"| PlasticityUsed="
                    f"{controller_plasticity:.4f} "
                    f"| PlasticityNow="
                    f"{measured_plasticity:.4f} "
                    f"| Mu="
                    f"{mu_client:.6f} "
                    f"| ReplayWeight="
                    f"{replay_weight:.4f} "
                    f"| ReplaySteps="
                    f"{stats['replay_steps']} "
                    f"| ReplaySamples="
                    f"{stats['replay_samples']}"
                )

                client_history.append(
                    [
                        stage_number,
                        round_idx + 1,
                        client_id + 1,
                        len(indices),
                        effective_samples,
                        stats["loss"],
                        stats["current_loss"],
                        stats["replay_loss"],
                        stats["prox_loss"],
                        raw_drift,
                        updated_ema,
                        stability_signal,
                        controller_plasticity,
                        measured_plasticity,
                        mu_client,
                        replay_weight,
                        stats["replay_steps"],
                        stats["replay_samples"]
                    ]
                )

                del local_model
                del client_loader
                del client_subset

                if device.type == "cuda":
                    torch.cuda.empty_cache()

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

        # ====================================================
        # Prototype Extraction
        # ====================================================

        print(
            "\nExtracting client-local "
            "current-stage prototypes..."
        )

        client_prototype_sets = []

        for client_id in range(
            args.clients
        ):

            indices = client_indices[
                client_id
            ]

            if len(indices) == 0:
                continue

            if (
                args.smoke_test
                and args.smoke_proto_samples
                is not None
            ):

                max_proto_samples = (
                    args.smoke_proto_samples
                )

            else:

                max_proto_samples = None

            client_bank = (
                extract_client_prototypes(
                    model=global_model,
                    dataset=stage_dataset,
                    indices=indices,
                    current_classes=set(
                        current_classes
                    ),
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                    workers=args.workers,
                    amp=args.amp,
                    prototypes_per_class=(
                        args.prototypes_per_class
                    ),
                    max_samples=(
                        max_proto_samples
                    )
                )
            )

            client_prototype_sets.append(
                client_bank
            )

            local_vectors = sum(
                len(v)
                for v in client_bank.values()
            )

            print(
                f"Client "
                f"{client_id + 1} "
                f"contributed "
                f"{local_vectors} prototypes."
            )

        prototype_bank = (
            merge_server_prototypes(
                prototype_bank,
                client_prototype_sets,
                args.prototypes_per_class
            )
        )

        (
            vectors,
            classes,
            mib,
            mb
        ) = prototype_bank_size(
            prototype_bank
        )

        print(
            "\nServer prototype bank "
            f"after S{stage_number}:"
        )

        print(
            f"  prototypes : {vectors}"
        )

        print(
            f"  classes    : {classes}"
        )

        print(
            f"  size       : "
            f"{mib:.3f} MiB "
            f"({mb:.3f} MB)"
        )

        memory_history.append(
            [
                stage_number,
                vectors,
                classes,
                mib,
                mb
            ]
        )

        # ====================================================
        # Save checkpoint
        # ====================================================

        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"e4v7_stage{stage_number}.pt"
        )

        torch.save(
            {
                "stage": stage_number,
                "model": (
                    global_model.state_dict()
                ),
                "prototype_bank": (
                    prototype_bank
                ),
                "mu_prox": (
                    args.mu_prox
                ),
                "client_drift_ema": (
                    client_drift_ema
                ),
                "client_last_drift": (
                    client_last_drift
                ),
                "client_last_plasticity": (
                    client_last_plasticity
                ),
                "drift_gamma": (
                    args.drift_gamma
                ),
                "drift_beta": (
                    args.drift_beta
                ),
                "drift_scale": (
                    args.drift_scale
                ),
                "mu_min": args.mu_min,
                "mu_max": args.mu_max,
                "args": vars(args)
            },
            checkpoint_path
        )

        print(
            "\nSaved checkpoint:",
            checkpoint_path
        )

        # ====================================================
        # Evaluation
        # ====================================================

        seen_classes = []

        for j in range(
            stage_idx + 1
        ):

            seen_classes.extend(
                stages[j]
            )

        print(
            f"\nEvaluating "
            f"{stage_idx + 1} seen stage(s)..."
        )

        for eval_idx in range(
            stage_idx + 1
        ):

            eval_classes = stages[
                eval_idx
            ]

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
                f"After S"
                f"{stage_number} "
                f"-> Test S"
                f"{eval_idx + 1}: "
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

            historical = (
                accuracy_matrix[
                    j:num_stages - 1,
                    j
                ]
            )

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

        avg_forgetting = float(
            "nan"
        )

        bwt = float(
            "nan"
        )

    (
        final_vectors,
        final_classes,
        final_mib,
        final_mb
    ) = prototype_bank_size(
        prototype_bank
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "E4-v7 FINAL RESULTS"
    )

    print(
        "=" * 80
    )

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

    # ========================================================
    # Save
    # ========================================================

    accuracy_file = os.path.join(
        args.results_dir,
        "e4v7_accuracy_matrix.csv"
    )

    f1_file = os.path.join(
        args.results_dir,
        "e4v7_f1_matrix.csv"
    )

    summary_file = os.path.join(
        args.results_dir,
        "e4v7_summary.csv"
    )

    client_file = os.path.join(
        args.results_dir,
        "e4v7_client_history.csv"
    )

    memory_file = os.path.join(
        args.results_dir,
        "e4v7_memory_history.csv"
    )

    drift_file = os.path.join(
        args.results_dir,
        "e4v7_controller_history.csv"
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
                "lambda_replay",
                "replay_batch",
                "mu_prox",
                "drift_gamma",
                "drift_beta",
                "drift_scale",
                "mu_min",
                "mu_max",
                "final_avg_accuracy",
                "final_avg_macro_f1",
                "avg_forgetting_pp",
                "bwt_pp",
                "prototype_memory_mib"
            ]
        )

        writer.writerow(
            [
                args.clients,
                args.alpha,
                num_stages,
                args.lambda_replay,
                args.replay_batch,
                args.mu_prox,
                args.drift_gamma,
                args.drift_beta,
                args.drift_scale,
                args.mu_min,
                args.mu_max,
                final_avg_accuracy,
                final_avg_f1,
                avg_forgetting,
                bwt,
                final_mib
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
                "loss",
                "current_loss",
                "replay_loss",
                "prox_loss",
                "prototype_functional_drift",
                "prototype_drift_ema",
                "stability_signal",
                "plasticity_used",
                "plasticity_measured",
                "adaptive_mu",
                "adaptive_replay_weight",
                "replay_steps",
                "replay_samples"
            ]
        )

        writer.writerows(
            client_history
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
                "prototypes",
                "classes",
                "memory_mib",
                "memory_mb"
            ]
        )

        writer.writerows(
            memory_history
        )

    trace_file = os.path.join(args.results_dir, "e10_controller_trace.csv")
    with open(trace_file, "w", newline="", encoding="utf-8") as tf:
        tw = csv.writer(tf)
        tw.writerow(["stage", "round", "client", "mode", "previous_raw_drift",
                     "previous_plasticity", "drift_scale", "stability_signal",
                     "plasticity_used", "lambda_preclip", "lambda_applied", "clipped"])
        tw.writerows(trace_rows)

    with open(
        drift_file,
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
                "prototype_functional_drift",
                "previous_prototype_drift_ema",
                "updated_prototype_drift_ema",
                "stability_signal",
                "plasticity_used",
                "plasticity_measured",
                "adaptive_mu",
                "adaptive_replay_weight"
            ]
        )

        writer.writerows(
            drift_history
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
        client_file
    )

    print(
        " ",
        memory_file
    )

    print(
        " ",
        drift_file
    )

    if args.smoke_test:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "E4-v7 SMOKE TEST COMPLETED"
        )

        print(
            "Check ProtoDrift/ProtoDriftEMA/DriftRatio and adaptive Mu; "
            "use at least 3 stages to observe causal adaptation."
        )

        print(
            "=" * 80
        )


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller-mode",
        choices=["full", "fixed", "drift_only", "plasticity_only", "fixed_125", "fixed_135", "fixed_150", "stage_ramp"],
        default="full",
        help="E8 replay-weight ablation; full reproduces original E4-v7 controller."
    )

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
        "--prototypes-per-class",
        type=int,
        default=10
    )

    parser.add_argument(
        "--lambda-replay",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--replay-lambda-min",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--replay-lambda-max",
        type=float,
        default=1.50
    )

    parser.add_argument(
        "--replay-batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--mu-prox",
        type=float,
        default=0.1,
        help="Base proximal coefficient; balanced E3-C reference is 0.1."
    )

    parser.add_argument(
        "--drift-gamma",
        type=float,
        default=25.0,
        help="Sensitivity of adaptive proximal strength to prototype functional drift."
    )

    parser.add_argument(
        "--drift-beta",
        type=float,
        default=0.9,
        help="EMA coefficient for persistent client drift."
    )

    parser.add_argument(
        "--drift-proto-samples",
        type=int,
        default=256,
        help="Maximum old prototypes used to estimate functional drift."
    )
    parser.add_argument(
        "--drift-scale",
        type=float,
        default=1000000.0,
        help="Scale in bounded mapping 1-exp(-drift_scale*prototype_drift)."
    )

    parser.add_argument(
        "--mu-min",
        type=float,
        default=0.02,
        help="Minimum adaptive proximal coefficient."
    )

    parser.add_argument(
        "--mu-max",
        type=float,
        default=0.5,
        help="Maximum adaptive proximal coefficient."
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
        "--smoke-proto-samples",
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
        default="checkpoints/e4v4"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/e4v4"
    )

    args = parser.parse_args()

    main(args)
