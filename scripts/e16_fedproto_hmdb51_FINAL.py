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
# HMDB51 Dataset (official split 1)
# ============================================================

class HMDBStageDataset(Dataset):

    def __init__(self, root, split_root, allowed_classes, split="train",
                 frames=8, size=224, train=False):
        self.root = root
        self.frames = frames
        self.size = size
        self.train = train
        self.allowed_classes = set(allowed_classes)

        split_files = sorted(
            f for f in os.listdir(split_root)
            if f.endswith("_test_split1.txt")
        )
        if len(split_files) != 51:
            raise RuntimeError(
                f"Expected 51 HMDB51 split-1 files in {split_root}, found {len(split_files)}"
            )

        class_names = [f[:-len("_test_split1.txt")] for f in split_files]
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        wanted_flag = 1 if split == "train" else 2
        self.samples = []

        for sf, class_name in zip(split_files, class_names):
            label = self.class_to_idx[class_name]
            if label not in self.allowed_classes:
                continue
            with open(os.path.join(split_root, sf), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    video_name, flag = parts[0], int(parts[1])
                    if flag != wanted_flag:
                        continue
                    full_path = os.path.join(root, class_name, class_name, video_name)
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
        indices = np.linspace(0, max(length - 1, 0), self.frames).astype(int)
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _transform(self, frames):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        do_flip = self.train and random.random() < 0.5
        output = []
        for frame in frames:
            x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            x = F.interpolate(x.unsqueeze(0), size=(self.size, self.size),
                              mode="bilinear", align_corners=False).squeeze(0)
            if do_flip:
                x = torch.flip(x, dims=[2])
            output.append((x - mean) / std)
        return torch.stack(output, dim=0)

    def __getitem__(self, index):
        path, label = self.samples[index]
        pixel_values = self._transform(self._read_video(path))
        return pixel_values, torch.tensor(label, dtype=torch.long)


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
# E16 FedProto-adapted utilities
# ============================================================

def clone_state_cpu(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def load_state_cpu(model, state):
    model.load_state_dict(state, strict=True)


def global_proto_tensor(global_prototypes, class_id, device, dtype):
    proto = global_prototypes.get(int(class_id))
    if proto is None:
        return None
    return proto.to(device=device, dtype=dtype)


def train_fedproto_client(model, loader, device, local_epochs, lr, weight_decay,
                          amp, global_prototypes, proto_lambda, max_batches=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    total_seen = 0
    total_loss = 0.0
    total_ce = 0.0
    total_proto = 0.0
    aligned = 0

    for epoch in range(local_epochs):
        model.train()
        bar = tqdm(loader, desc=f"FedProto local epoch {epoch+1}/{local_epochs}", leave=False)
        for batch_idx, (videos, labels) in enumerate(bar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                outputs = model.timesformer(pixel_values=videos)
                features = outputs.last_hidden_state[:, 0, :]
                logits = model.classifier(features)
                ce_loss = ce(logits, labels)

                proto_terms = []
                for i, y in enumerate(labels):
                    gp = global_proto_tensor(global_prototypes, int(y.item()), device, features.dtype)
                    if gp is not None:
                        proto_terms.append(mse(features[i], gp))
                if proto_terms:
                    proto_loss = torch.stack(proto_terms).mean()
                    aligned += len(proto_terms)
                else:
                    proto_loss = torch.zeros((), device=device, dtype=features.dtype)
                loss = ce_loss + float(proto_lambda) * proto_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            total_seen += bs
            total_loss += float(loss.detach().item()) * bs
            total_ce += float(ce_loss.detach().item()) * bs
            total_proto += float(proto_loss.detach().item()) * bs
            bar.set_postfix(loss=total_loss/max(total_seen,1), ce=total_ce/max(total_seen,1), proto=total_proto/max(total_seen,1))

    return {
        "loss": total_loss/max(total_seen,1),
        "ce_loss": total_ce/max(total_seen,1),
        "proto_loss": total_proto/max(total_seen,1),
        "aligned_samples": aligned,
        "seen_samples": total_seen,
    }


@torch.no_grad()
def extract_class_centroids(model, dataset, indices, current_classes, device,
                            batch_size, workers, amp, max_samples=None):
    work = list(indices)
    if max_samples is not None:
        work = work[:min(max_samples, len(work))]
    if not work:
        return {}, {}
    loader = DataLoader(Subset(dataset, work), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=(workers > 0))
    model.eval()
    sums = {}
    counts = defaultdict(int)
    for videos, labels in tqdm(loader, desc="FedProto centroid extraction", leave=False):
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            outputs = model.timesformer(pixel_values=videos)
            feats = outputs.last_hidden_state[:, 0, :].float()
        for feat, label in zip(feats, labels):
            c = int(label.item())
            if c not in current_classes:
                continue
            f = feat.detach().cpu()
            sums[c] = f.clone() if c not in sums else sums[c] + f
            counts[c] += 1
    centroids = {c: (sums[c] / float(counts[c])).float() for c in sums}
    return centroids, dict(counts)


def aggregate_fedproto(client_centroids, client_counts, old_global=None):
    # Original FedProto principle: class-wise aggregation of client prototypes.
    # Preserve old-class prototypes in the continual adaptation when a class is absent.
    out = {} if old_global is None else {c: p.clone().cpu().float() for c,p in old_global.items()}
    classes = set()
    for d in client_centroids:
        classes.update(d.keys())
    for c in classes:
        num = None
        den = 0.0
        for protos, counts in zip(client_centroids, client_counts):
            if c not in protos:
                continue
            w = float(counts.get(c, 1))
            term = protos[c].float() * w
            num = term.clone() if num is None else num + term
            den += w
        if num is not None and den > 0:
            out[c] = (num / den).cpu().float()
    return out


@torch.no_grad()
def evaluate_model(model, loader, device, seen_classes, eval_classes, amp, num_labels=51):
    model.eval()
    truth, pred = [], []
    unseen = torch.ones(num_labels, dtype=torch.bool, device=device)
    unseen[list(seen_classes)] = False
    for videos, labels in tqdm(loader, desc="Eval", leave=False):
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(pixel_values=videos).logits
        logits[:, unseen] = -1e9
        p = logits.argmax(dim=1).cpu().numpy()
        truth.extend(labels.numpy())
        pred.extend(p)
    return (accuracy_score(truth, pred)*100.0,
            f1_score(truth, pred, labels=sorted(eval_classes), average="macro", zero_division=0)*100.0)


def mean_client_eval(client_states, model_factory, loader, device, seen_classes, eval_classes, amp):
    accs, f1s = [], []
    for cid, state in sorted(client_states.items()):
        model = model_factory()
        load_state_cpu(model, state)
        model.to(device)
        a, f = evaluate_model(model, loader, device, seen_classes, eval_classes, amp)
        accs.append(a); f1s.append(f)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    return float(np.mean(accs)), float(np.mean(f1s)), accs, f1s


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    stages = [list(range(0,10)), list(range(10,20)), list(range(20,30)),
              list(range(30,40)), list(range(40,51))]
    if args.max_stages is not None:
        stages = stages[:args.max_stages]

    print("="*80)
    print("E16 FedProto-adapted | HMDB51 continual federated video baseline")
    print("="*80)
    print("Device          :", device)
    if device.type == "cuda": print("GPU             :", torch.cuda.get_device_name(0))
    print("Clients         :", args.clients)
    print("Dirichlet alpha :", args.alpha)
    print("Seed            :", args.seed)
    print("Proto lambda    :", args.proto_lambda)
    print("IMPORTANT       : no FedAvg/model-parameter aggregation")
    print("Evaluation      : unweighted mean of client-model metrics on common stage test sets")

    def make_model():
        m = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400", num_labels=51,
            ignore_mismatched_sizes=True)
        configure_partial_finetuning(m, args.last_blocks)
        return m

    # Identical K400 initialization for every client. Persistent thereafter.
    base = make_model()
    base_state = clone_state_cpu(base)
    del base
    client_states = {cid: OrderedDict((k,v.clone()) for k,v in base_state.items())
                     for cid in range(args.clients)}
    global_prototypes = {}
    nst = len(stages)
    acc_mat = np.full((nst,nst), np.nan)
    f1_mat = np.full((nst,nst), np.nan)
    history = []
    proto_history = []

    for sidx, current_classes in enumerate(stages):
        sn = sidx + 1
        print("\n" + "="*80)
        print(f"FEDPROTO STAGE {sn}/{nst}: classes {current_classes[0]+1}-{current_classes[-1]+1}")
        print("="*80)
        ds = HMDBStageDataset(args.data_root, args.split_root, current_classes, split="train",
                              frames=args.frames, size=args.image_size, train=True)
        parts = dirichlet_partition(ds, args.clients, args.alpha, args.seed+sidx)
        print("Stage training samples:", len(ds))
        print("Client partitions:", [len(x) for x in parts])

        for rnd in range(args.rounds_per_stage):
            local_centroids, local_counts = [], []
            for cid in range(args.clients):
                idx = parts[cid]
                if not idx:
                    continue
                print(f"\nClient {cid+1}/{args.clients} | samples={len(idx)}")
                model = make_model()
                load_state_cpu(model, client_states[cid])
                model.to(device)
                loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True,
                                    persistent_workers=(args.workers>0))
                stats = train_fedproto_client(
                    model, loader, device, args.local_epochs, args.lr, args.weight_decay,
                    args.amp, global_prototypes, args.proto_lambda, args.max_local_batches)
                client_states[cid] = clone_state_cpu(model)
                maxp = args.smoke_proto_samples if args.smoke_test else None
                cent, cnt = extract_class_centroids(
                    model, ds, idx, set(current_classes), device, args.batch_size,
                    args.workers, args.amp, maxp)
                local_centroids.append(cent); local_counts.append(cnt)
                history.append([sn,rnd+1,cid+1,len(idx),stats["loss"],stats["ce_loss"],
                                stats["proto_loss"],stats["aligned_samples"],stats["seen_samples"]])
                print(f"Client {cid+1} | loss={stats['loss']:.4f} | CE={stats['ce_loss']:.4f} | Proto={stats['proto_loss']:.6f} | aligned={stats['aligned_samples']}")
                del model, loader
                if device.type == "cuda": torch.cuda.empty_cache()

            global_prototypes = aggregate_fedproto(local_centroids, local_counts, global_prototypes)
            print(f"\nFedProto server aggregation complete: {len(global_prototypes)} class prototypes")

        bytes_used = sum(p.numel()*p.element_size() for p in global_prototypes.values())
        mib = bytes_used/(1024**2)
        proto_history.append([sn,len(global_prototypes),mib])
        ckpt = os.path.join(args.checkpoint_dir, f"e16_hmdb51_fedproto_stage{sn}.pt")
        torch.save({"stage":sn,"client_states":client_states,"global_prototypes":global_prototypes,
                    "args":vars(args)}, ckpt)
        print("Saved checkpoint:", ckpt)

        seen = [c for j in range(sidx+1) for c in stages[j]]
        print(f"\nEvaluating {sidx+1} seen stage(s) across {args.clients} persistent client models...")
        for eidx in range(sidx+1):
            ec = stages[eidx]
            tds = UCFStageDataset(args.data_root, test_file, class_file, ec,
                                  frames=args.frames, size=args.image_size, train=False)
            if args.smoke_test and args.smoke_eval_samples is not None:
                tds = Subset(tds, list(range(min(args.smoke_eval_samples,len(tds)))))
            tl = DataLoader(tds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                            pin_memory=True,persistent_workers=(args.workers>0))
            a,f,_,_ = mean_client_eval(client_states, make_model, tl, device, seen, ec, args.amp)
            acc_mat[sidx,eidx]=a; f1_mat[sidx,eidx]=f
            print(f"After S{sn} -> Test S{eidx+1}: MeanClient Acc={a:.3f}% F1={f:.3f}%")

    final_acc=float(np.nanmean(acc_mat[-1])); final_f1=float(np.nanmean(f1_mat[-1]))
    forgetting=[]
    for j in range(nst-1):
        best=float(np.nanmax(acc_mat[j:nst-1,j])); forgetting.append(best-float(acc_mat[-1,j]))
    avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0
    bwt=float(np.mean([acc_mat[-1,j]-acc_mat[j,j] for j in range(nst-1)])) if nst>1 else 0.0

    print("\n"+"="*80)
    print("E16 FEDPROTO-ADAPTED FINAL RESULTS")
    print("="*80)
    print(f"Final Average Accuracy : {final_acc:.3f}%")
    print(f"Final Average Macro-F1 : {final_f1:.3f}%")
    print(f"Average Forgetting     : {avg_forgetting:.3f} pp")
    print(f"Backward Transfer      : {bwt:.3f} pp")
    print(f"Global class prototypes: {len(global_prototypes)}")

    np.savetxt(os.path.join(args.results_dir,"e16_hmdb51_accuracy_matrix.csv"),acc_mat,delimiter=",")
    np.savetxt(os.path.join(args.results_dir,"e16_hmdb51_f1_matrix.csv"),f1_mat,delimiter=",")
    with open(os.path.join(args.results_dir,"e16_hmdb51_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["clients","alpha","seed","proto_lambda","final_avg_accuracy","final_avg_f1","avg_forgetting","bwt","prototype_classes"])
        w.writerow([args.clients,args.alpha,args.seed,args.proto_lambda,final_acc,final_f1,avg_forgetting,bwt,len(global_prototypes)])
    with open(os.path.join(args.results_dir,"e16_hmdb51_client_history.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["stage","round","client","partition_samples","loss","ce_loss","proto_loss","aligned_samples","seen_samples"]); w.writerows(history)
    with open(os.path.join(args.results_dir,"e16_hmdb51_prototype_history.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["stage","classes","memory_mib"]); w.writerows(proto_history)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True)
    p.add_argument("--split-root",required=True)
    p.add_argument("--frames",type=int,default=8)
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--clients",type=int,default=2)
    p.add_argument("--alpha",type=float,default=0.5)
    p.add_argument("--rounds-per-stage",type=int,default=1)
    p.add_argument("--local-epochs",type=int,default=1)
    p.add_argument("--last-blocks",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--weight-decay",type=float,default=0.05)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--amp",action="store_true")
    p.add_argument("--proto-lambda",type=float,default=1.0,
                   help="FedProto prototype-alignment coefficient (official code option --ld; default 1).")
    p.add_argument("--max-stages",type=int,default=None)
    p.add_argument("--max-local-batches",type=int,default=None)
    p.add_argument("--smoke-eval-samples",type=int,default=40)
    p.add_argument("--smoke-proto-samples",type=int,default=40)
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--checkpoint-dir",default="checkpoints/e16_fedproto_hmdb51")
    p.add_argument("--results-dir",default="results/e16_fedproto_hmdb51")
    main(p.parse_args())
