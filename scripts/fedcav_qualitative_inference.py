"""FedCAV matched-checkpoint qualitative inference; run from C:\\Users\\abc\\FedCAV.
Requires the original scripts/e4v7_fixed.py and trusted locally produced checkpoints.
"""
import argparse
import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import TimesformerConfig, TimesformerForVideoClassification


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='.')
    p.add_argument('--data-root', default=r'C:\Users\abc\ST-EdgeViT\datasets\UCF101')
    p.add_argument('--split-root', default=r'C:\Users\abc\FedCAV\data\ucfTrainTestlist')
    p.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    p.add_argument('--stage', type=int, choices=[1,2,3,4,5], default=5)
    p.add_argument('--limit', type=int, default=0, help='0 = all eligible test clips')
    p.add_argument('--output', default='results/qualitative')
    return p.parse_args()


def main():
    a = get_args()
    project = Path(a.project).resolve()
    src = project / 'scripts' / 'e4v7_fixed.py'
    if not src.is_file():
        src = project / 'e4v7_fixed.py'
    if not src.is_file():
        raise FileNotFoundError('Original e4v7_fixed.py not found in scripts/ or project root')
    spec = importlib.util.spec_from_file_location('fedcav_training', src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    adir = {42:'e4v7_full',43:'e4v7_adaptive_seed43',44:'e4v7_adaptive_seed44'}[a.seed]
    fdir = {42:'e4v7_fixedlambda_full',43:'e4v7_fixedlambda_seed43',44:'e4v7_fixedlambda_seed44'}[a.seed]
    paths = [project/'checkpoints'/folder/f'e4v7_stage{a.stage}.pt' for folder in (fdir,adir)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    configs = []
    for path in paths:
        # Only load checkpoints created by you: torch.load can execute pickled code.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if int(checkpoint['stage']) != a.stage:
            raise ValueError(f'Checkpoint stage mismatch: {path}')
        cfg_args = checkpoint.get('args', {})
        configs.append(cfg_args)
        cfg = TimesformerConfig.from_pretrained('facebook/timesformer-base-finetuned-k400')
        cfg.num_labels = 101
        model = TimesformerForVideoClassification(cfg)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device).eval()
        models.append(model)
        del checkpoint
    frames = int(configs[0].get('frames',8))
    size = int(configs[0].get('image_size',224))
    if (frames,size) != (int(configs[1].get('frames',8)),int(configs[1].get('image_size',224))):
        raise ValueError('Fixed/adaptive checkpoint input settings differ')
    class_file = Path(a.split_root)/'classInd.txt'
    names = {}
    for line in class_file.read_text(encoding='utf-8').splitlines():
        idx,name = line.strip().split(maxsplit=1)
        names[int(idx)-1] = name
    stages = [list(range(i,min(i+20,101))) for i in (0,20,40,60,80)]
    seen = set(x for group in stages[:a.stage] for x in group)
    ds = mod.UCFStageDataset(str(Path(a.data_root)), str(Path(a.split_root)/'testlist01.txt'),
                             str(class_file), sorted(seen), frames=frames,size=size,train=False)
    if not ds.samples:
        raise RuntimeError('No test clips found; verify data-root and split-root')
    output = project / a.output / f'seed{a.seed}_stage{a.stage}'
    output.mkdir(parents=True, exist_ok=True)
    csvpath = output/'matched_predictions.csv'
    errors = output/'decode_errors.csv'
    categories = Counter()
    error_rows=[]
    unseen = torch.tensor(sorted(set(range(101))-seen), dtype=torch.long, device=device)
    count = min(len(ds),a.limit) if a.limit>0 else len(ds)
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['seed','stage','sample_index','video_path','ground_truth_id','ground_truth',
            'fixed_prediction_id','fixed_prediction','fixed_confidence','adaptive_prediction_id','adaptive_prediction',
            'adaptive_confidence','fixed_correct','adaptive_correct','category'])
        writer.writeheader()
        for i in range(count):
            path,label = ds.samples[i]
            try:
                video,_ = ds[i]
            except Exception as exc:
                error_rows.append((i,path,str(exc)))
                continue
            video = video.unsqueeze(0).to(device)
            preds=[]
            with torch.inference_mode():
                for model in models:
                    with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                        logits=model(pixel_values=video).logits.float()
                    if unseen.numel():
                        logits[:,unseen] = -1e9
                    probs=logits.softmax(dim=-1)
                    conf, pred=probs.max(dim=-1)
                    preds.append((int(pred.item()),float(conf.item())))
            (fp,fc),(ap,ac)=preds
            fixed_ok=fp==label
            adaptive_ok=ap==label
            category=('both_correct' if fixed_ok and adaptive_ok else
                      'adaptive_only_correct' if adaptive_ok else
                      'fixed_only_correct' if fixed_ok else 'both_incorrect')
            categories[category]+=1
            writer.writerow(dict(seed=a.seed,stage=a.stage,sample_index=i,video_path=path,
                ground_truth_id=label,ground_truth=names[label],fixed_prediction_id=fp,
                fixed_prediction=names[fp],fixed_confidence=f'{fc:.6f}',
                adaptive_prediction_id=ap,adaptive_prediction=names[ap],adaptive_confidence=f'{ac:.6f}',
                fixed_correct=int(fixed_ok),adaptive_correct=int(adaptive_ok),category=category))
            if (i+1)%100==0:
                print(f'{i+1}/{count} evaluated; {dict(categories)}',flush=True)
    with errors.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_index','video_path','error'])
        writer.writerows(error_rows)
    print('Saved:',csvpath)
    print('Decode errors:',len(error_rows),'(see',errors,')')
    print('Categories:',dict(categories))
    print('NOTE: Confidence is softmax over classes seen by the evaluated stage.')

if __name__=='__main__':
    main()
