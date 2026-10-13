$ErrorActionPreference = "Stop"
Set-Location "C:\Users\abc\FedCAV"
$DATA_ROOT = "C:\Users\abc\ST-EdgeViT\datasets\UCF101"
$SPLIT_ROOT = "C:\Users\abc\FedCAV\data\ucfTrainTestlist"
if (!(Test-Path $DATA_ROOT) -or !(Test-Path $SPLIT_ROOT)) { throw "Set DATA_ROOT and SPLIT_ROOT to your actual dataset paths." }
foreach ($mode in @("fixed_125", "fixed_135", "fixed_150", "stage_ramp", "full")) {
  python scripts/e9_e10_fedcav.py --controller-mode $mode --data-root $DATA_ROOT --split-root $SPLIT_ROOT --clients 2 --alpha 0.5 --rounds-per-stage 1 --local-epochs 1 --batch-size 2 --frames 8 --seed 42 --mu-prox 0.001 --lambda-replay 1.0 --replay-lambda-min 0.75 --replay-lambda-max 1.5 --drift-scale 1000000 --checkpoint-dir "checkpoints/e9_$($mode)_seed42" --results-dir "results/e9_$($mode)_seed42"
  if ($LASTEXITCODE -ne 0) { throw "Run failed: $mode" }
}
