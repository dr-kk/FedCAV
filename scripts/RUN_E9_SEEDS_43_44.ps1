
$ErrorActionPreference = "Stop"

$PROJECT_ROOT = "C:\Users\abc\FedCAV"
$DATA_ROOT = "C:\Users\abc\ST-EdgeViT\datasets\UCF101"
$SPLIT_ROOT = "C:\Users\abc\FedCAV\data\ucfTrainTestlist"
$PYTHON_SCRIPT = Join-Path $PROJECT_ROOT "scripts\e9_e10_fedcav.py"

Set-Location $PROJECT_ROOT

# Verify dataset and experiment script.
foreach ($path in @(
    $DATA_ROOT,
    $SPLIT_ROOT,
    $PYTHON_SCRIPT,
    (Join-Path $SPLIT_ROOT "trainlist01.txt"),
    (Join-Path $SPLIT_ROOT "testlist01.txt")
)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

python -m py_compile $PYTHON_SCRIPT
if ($LASTEXITCODE -ne 0) {
    throw "Python syntax check failed."
}

# Repeat the two comparison arms on additional seeds.
foreach ($seed in @(43, 44)) {
    foreach ($mode in @("fixed_150", "full")) {

        $checkpointDir = "checkpoints/e9_$($mode)_seed$seed"
        $resultsDir = "results/e9_$($mode)_seed$seed"

        # Never overwrite a previous or partially completed run.
        if ((Test-Path $checkpointDir) -or (Test-Path $resultsDir)) {
            throw "Output already exists for mode=$mode seed=$seed. Inspect it before rerunning."
        }

        Write-Host ""
        Write-Host "========================================"
        Write-Host "Starting E9: mode=$mode, seed=$seed"
        Write-Host "========================================"

        python $PYTHON_SCRIPT `
            --controller-mode $mode `
            --data-root $DATA_ROOT `
            --split-root $SPLIT_ROOT `
            --clients 2 `
            --alpha 0.5 `
            --rounds-per-stage 1 `
            --local-epochs 1 `
            --batch-size 2 `
            --frames 8 `
            --seed $seed `
            --mu-prox 0.001 `
            --lambda-replay 1.0 `
            --replay-lambda-min 0.75 `
            --replay-lambda-max 1.5 `
            --drift-scale 1000000 `
            --checkpoint-dir $checkpointDir `
            --results-dir $resultsDir

        if ($LASTEXITCODE -ne 0) {
            throw "Run failed: mode=$mode seed=$seed"
        }

        $summary = Join-Path $resultsDir "e4v7_summary.csv"

        if (-not (Test-Path $summary)) {
            throw "Run finished without summary CSV: $summary"
        }

        Write-Host "Completed: mode=$mode seed=$seed"
    }
}

Write-Host ""
Write-Host "All four E9 replication runs completed."