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
                    # Same path convention as the successfully completed E16 HMDB51 run.
                    # One locally absent/unreadable official-split clip is excluded explicitly.
                    if video_name == 'torwart_de-Camp_2009-_Training_hoher_B_lle_mit_Airbodies_catch_f_cm_np1_fr_bad_2.avi':
                        continue
                    full_path = os.path.join(root, class_name, class_name, video_name)

                    # Validate the clip with the same OpenCV backend used by __getitem__.
                    # HMDB51 split metadata can reference clips that are absent/unreadable
                    # in a local extraction. Exclude such clips before client partitioning
                    # so training/evaluation never changes samples dynamically.
                    cap = cv2.VideoCapture(full_path)
                    readable = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
                    cap.release()
                    if not readable:
                        continue

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
# E17 FedProx utilities
# ============================================================

def clone_state_cpu(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def load_state_cpu(model, state):
    model.load_state_dict(state, strict=True)


def train_fedprox_client(model, loader, device, local_epochs, lr, weight_decay,
                         amp, global_state, mu, max_batches=None):
    """FedProx local objective: CE + (mu/2)||w-w_global||^2 on trainable parameters."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    refs = {n: global_state[n].to(device=device, dtype=p.dtype) for n,p in trainable}
    total_seen=0; total_loss=0.0; total_ce=0.0; total_prox=0.0
    for epoch in range(local_epochs):
        model.train()
        bar=tqdm(loader, desc=f"FedProx local epoch {epoch+1}/{local_epochs}", leave=False)
        for batch_idx,(videos,labels) in enumerate(bar):
            if max_batches is not None and batch_idx >= max_batches: break
            videos=videos.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type=="cuda")):
                logits=model(pixel_values=videos).logits
                ce_loss=ce(logits,labels)
                prox=torch.zeros((),device=device,dtype=ce_loss.dtype)
                for n,p in trainable:
                    prox = prox + torch.sum((p-refs[n])**2)
                loss=ce_loss + 0.5*float(mu)*prox
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            bs=labels.size(0); total_seen+=bs
            total_loss+=float(loss.detach().item())*bs
            total_ce+=float(ce_loss.detach().item())*bs
            total_prox+=float(prox.detach().item())*bs
            bar.set_postfix(loss=total_loss/max(total_seen,1),ce=total_ce/max(total_seen,1),prox=total_prox/max(total_seen,1))
    return {"loss":total_loss/max(total_seen,1),"ce_loss":total_ce/max(total_seen,1),
            "prox_term":total_prox/max(total_seen,1),"seen_samples":total_seen}


def weighted_fedavg(local_states, weights):
    if not local_states: raise RuntimeError("No client states available for FedAvg")
    total=float(sum(weights)); out=OrderedDict()
    for k in local_states[0].keys():
        ref=local_states[0][k]
        if not torch.is_floating_point(ref):
            out[k]=ref.clone(); continue
        acc=torch.zeros_like(ref,dtype=torch.float32)
        for st,w in zip(local_states,weights): acc.add_(st[k].float(),alpha=float(w)/total)
        out[k]=acc.to(dtype=ref.dtype)
    return out


@torch.no_grad()
def evaluate_global_state(global_state, model_factory, loader, device, seen_classes, eval_classes, amp):
    model=model_factory(); load_state_cpu(model,global_state); model.to(device)
    a,f,yt,yp=evaluate_model(model,loader,device,seen_classes,eval_classes,amp)
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return a,f,yt,yp


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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir,exist_ok=True); os.makedirs(args.checkpoint_dir,exist_ok=True)
    stages=[list(range(0,10)),list(range(10,20)),list(range(20,30)),list(range(30,40)),list(range(40,51))]
    if args.max_stages is not None: stages=stages[:args.max_stages]
    print("="*80); print("E17 FedProx | HMDB51 continual federated video baseline"); print("="*80)
    print("Device          :",device)
    if device.type=="cuda": print("GPU             :",torch.cuda.get_device_name(0))
    print("Clients         :",args.clients); print("Dirichlet alpha :",args.alpha); print("Seed            :",args.seed)
    print("FedProx mu      :",args.mu); print("Aggregation     : sample-weighted FedAvg after each round")
    print("Evaluation      : global server model on common stage test sets")
    def make_model():
        m=TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400",num_labels=51,ignore_mismatched_sizes=True)
        configure_partial_finetuning(m,args.last_blocks); return m
    base=make_model(); global_state=clone_state_cpu(base); del base
    nst=len(stages); acc_mat=np.full((nst,nst),np.nan); f1_mat=np.full((nst,nst),np.nan); history=[]
    for sidx,current_classes in enumerate(stages):
        sn=sidx+1; print("\n"+"="*80); print(f"FEDPROX STAGE {sn}/{nst}: classes {current_classes[0]+1}-{current_classes[-1]+1}"); print("="*80)
        ds=HMDBStageDataset(args.data_root,args.split_root,current_classes,split="train",frames=args.frames,size=args.image_size,train=True)
        parts=dirichlet_partition(ds,args.clients,args.alpha,args.seed+sidx)
        print("Stage training samples:",len(ds)); print("Client partitions:",[len(x) for x in parts])
        for rnd in range(args.rounds_per_stage):
            local_states=[]; local_weights=[]
            round_global=OrderedDict((k,v.clone()) for k,v in global_state.items())
            for cid in range(args.clients):
                idx=parts[cid]
                if not idx: continue
                print(f"\nClient {cid+1}/{args.clients} | samples={len(idx)}")
                model=make_model(); load_state_cpu(model,round_global); model.to(device)
                loader=DataLoader(Subset(ds,idx),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=(args.workers>0))
                stats=train_fedprox_client(model,loader,device,args.local_epochs,args.lr,args.weight_decay,args.amp,round_global,args.mu,args.max_local_batches)
                local_states.append(clone_state_cpu(model)); local_weights.append(len(idx))
                history.append([sn,rnd+1,cid+1,len(idx),stats["loss"],stats["ce_loss"],stats["prox_term"],stats["seen_samples"]])
                print(f"Client {cid+1} | loss={stats['loss']:.4f} | CE={stats['ce_loss']:.4f} | Prox={stats['prox_term']:.6f}")
                del model,loader
                if device.type=="cuda": torch.cuda.empty_cache()
            global_state=weighted_fedavg(local_states,local_weights)
            print("\nFedProx server aggregation complete")
        ckpt=os.path.join(args.checkpoint_dir,f"e17_hmdb51_fedprox_stage{sn}.pt")
        if args.smoke_test: print("Smoke test: checkpoint save skipped")
        else:
            torch.save({"stage":sn,"global_state":global_state,"args":vars(args)},ckpt); print("Saved checkpoint:",ckpt)
        seen=[c for j in range(sidx+1) for c in stages[j]]
        print(f"\nEvaluating global server model on {sidx+1} seen stage(s)...")
        for eidx in range(sidx+1):
            ec=stages[eidx]; tds=HMDBStageDataset(args.data_root,args.split_root,ec,split="test",frames=args.frames,size=args.image_size,train=False)
            if args.smoke_test and args.smoke_eval_samples is not None: tds=Subset(tds,list(range(min(args.smoke_eval_samples,len(tds)))))
            tl=DataLoader(tds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=(args.workers>0))
            a,f,_,_=evaluate_global_state(global_state,make_model,tl,device,seen,ec,args.amp)
            acc_mat[sidx,eidx]=a; f1_mat[sidx,eidx]=f; print(f"After S{sn} -> Test S{eidx+1}: Global Acc={a:.3f}% F1={f:.3f}%")
    final_acc=float(np.nanmean(acc_mat[-1])); final_f1=float(np.nanmean(f1_mat[-1])); forgetting=[]
    for j in range(nst-1):
        best=float(np.nanmax(acc_mat[j:nst-1,j])); forgetting.append(best-float(acc_mat[-1,j]))
    avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0
    bwt=float(np.mean([acc_mat[-1,j]-acc_mat[j,j] for j in range(nst-1)])) if nst>1 else 0.0
    print("\n"+"="*80); print("E17 FEDPROX FINAL RESULTS"); print("="*80)
    print(f"Final Average Accuracy : {final_acc:.3f}%"); print(f"Final Average Macro-F1 : {final_f1:.3f}%")
    print(f"Average Forgetting     : {avg_forgetting:.3f} pp"); print(f"Backward Transfer      : {bwt:.3f} pp")
    np.savetxt(os.path.join(args.results_dir,"e17_hmdb51_accuracy_matrix.csv"),acc_mat,delimiter=",")
    np.savetxt(os.path.join(args.results_dir,"e17_hmdb51_f1_matrix.csv"),f1_mat,delimiter=",")
    with open(os.path.join(args.results_dir,"e17_hmdb51_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["clients","alpha","seed","mu","final_avg_accuracy","final_avg_f1","avg_forgetting","bwt"]); w.writerow([args.clients,args.alpha,args.seed,args.mu,final_acc,final_f1,avg_forgetting,bwt])
    with open(os.path.join(args.results_dir,"e17_hmdb51_client_history.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["stage","round","client","partition_samples","loss","ce_loss","prox_term","seen_samples"]); w.writerows(history)


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
    p.add_argument("--mu",type=float,default=0.01,help="FedProx proximal coefficient; default 0.01.")
    p.add_argument("--max-stages",type=int,default=None)
    p.add_argument("--max-local-batches",type=int,default=None)
    p.add_argument("--smoke-eval-samples",type=int,default=40)
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--checkpoint-dir",default="D:/FedCAV_Results/E17_HMDB51/checkpoints")
    p.add_argument("--results-dir",default="D:/FedCAV_Results/E17_HMDB51/results_seed42")
    main(p.parse_args())
