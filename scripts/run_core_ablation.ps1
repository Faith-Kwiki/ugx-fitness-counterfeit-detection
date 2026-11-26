param(
    [int]$Seed = 42,
    [int]$Epochs = 15,
    [int]$BatchSize = 8,
    [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Run this script from the repository root after creating .venv."
}

$cudaAvailable = & $pythonExe -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cudaAvailable.Trim() -ne "True") {
    throw "CUDA is not available. Install a compatible CUDA-enabled PyTorch build first."
}

$datasetCandidates = @(
    Get-ChildItem -LiteralPath "data\raw" -Directory |
        ForEach-Object {
            Join-Path $_.FullName "labelled_ugandan_banknote_dataset_clean"
        } |
        Where-Object { Test-Path -LiteralPath $_ }
)
if ($datasetCandidates.Count -ne 1) {
    throw "Expected exactly one labelled dataset folder under data\raw; found $($datasetCandidates.Count)."
}
$datasetDir = $datasetCandidates[0]
foreach ($split in "train_set", "validation_set", "test_set") {
    foreach ($className in "counterfeit", "genuine") {
        $classPath = Join-Path $datasetDir "splits\$split\$className"
        if (-not (Test-Path -LiteralPath $classPath)) {
            throw "Required dataset folder is missing: $classPath"
        }
    }
}

$runStamp = Get-Date -Format "yyyyMMddTHHmmss"
$cacheDir = "data\processed\streams_v1_224"
$runRoot = "runs\multisignal"
$experiments = @(
    @{ Name = "ablation-rgb"; Signals = @("rgb") },
    @{ Name = "ablation-rgb-edges"; Signals = @("rgb", "edges") },
    @{ Name = "ablation-rgb-texture"; Signals = @("rgb", "texture") },
    @{ Name = "ablation-rgb-edges-texture"; Signals = @("rgb", "edges", "texture") }
)
foreach ($experiment in $experiments) {
    $runId = "$($experiment.Name)-seed$Seed-$runStamp"
    $runPath = Join-Path $runRoot $runId
    if (Test-Path -LiteralPath $runPath) {
        throw "Run folder already exists; use a new run stamp: $runPath"
    }
    $experiment.RunId = $runId
}

Write-Host "CUDA device:"
& $pythonExe -c "import torch; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the CUDA device."
}

Write-Host "Preparing train and validation streams. The test split is not cached."
& $pythonExe -m src.banknote.preprocessing.cache_streams --dataset-dir $datasetDir --cache-dir $cacheDir
if ($LASTEXITCODE -ne 0) {
    throw "Preprocessing cache creation failed."
}

$learningRateText = $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture)
foreach ($experiment in $experiments) {
    Write-Host ""
    Write-Host "Starting $($experiment.RunId) with signals: $($experiment.Signals -join ', ')"
    $trainArgs = @(
        "-m",
        "src.banknote.training.train_multisignal",
        "--dataset-dir", $datasetDir,
        "--cache-dir", $cacheDir,
        "--run-id", $experiment.RunId,
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--learning-rate", $learningRateText,
        "--signals"
    ) + $experiment.Signals
    & $pythonExe @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $($experiment.RunId). Earlier run folders are preserved."
    }
}

Write-Host ""
Write-Host "Core concatenation ablation complete. Check runs\multisignal for each run's artifacts."
Write-Host "No test-set scores were calculated."
