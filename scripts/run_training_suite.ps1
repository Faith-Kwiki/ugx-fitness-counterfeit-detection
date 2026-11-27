param(
    [int]$Seed = 42,
    [int]$Epochs = 15,
    [int]$BatchSize = 8,
    [string]$LearningRate = "0.001"
)

$ErrorActionPreference = "Stop"

if ($Epochs -lt 1) {
    throw "Epochs must be greater than zero."
}
if ($BatchSize -lt 1) {
    throw "Batch size must be greater than zero."
}
$ParsedLearningRate = 0.0
if (-not [double]::TryParse(
    $LearningRate,
    [Globalization.NumberStyles]::Float,
    [Globalization.CultureInfo]::InvariantCulture,
    [ref]$ParsedLearningRate
) -or $ParsedLearningRate -le 0) {
    throw "Learning rate must be a positive number, such as 0.001."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Missing .venv\Scripts\python.exe. Create and install the project environment first."
}

function Invoke-ProjectPython {
    param([Parameter(Mandatory)][string[]]$PythonArguments)

    & $PythonExe @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

$CudaAvailable = & $PythonExe -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Could not check CUDA. Confirm PyTorch is installed in .venv."
}
if (($CudaAvailable | Select-Object -Last 1).Trim() -ne "True") {
    throw "CUDA is not available. Install a CUDA-enabled PyTorch build for this NVIDIA GPU first."
}

$RawDataRoot = Join-Path $ProjectRoot "data\raw"
$DatasetCandidates = @(
    Get-ChildItem -LiteralPath $RawDataRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Get-ChildItem -LiteralPath $_.FullName -Directory -Filter "labelled_ugandan_banknote_dataset_clean" -ErrorAction SilentlyContinue
        }
)
if ($DatasetCandidates.Count -ne 1) {
    throw "Expected exactly one labelled dataset folder two levels under data\raw; found $($DatasetCandidates.Count)."
}
$DatasetDir = $DatasetCandidates[0].FullName

Invoke-ProjectPython -PythonArguments @(
    "-m", "src.banknote.validation.validate_splits",
    "--dataset-dir", $DatasetDir
)

foreach ($Split in @("train_set", "validation_set", "test_set")) {
    foreach ($ClassName in @("counterfeit", "genuine")) {
        $ClassPath = Join-Path $DatasetDir "splits\$Split\$ClassName"
        if (-not (Test-Path -LiteralPath $ClassPath -PathType Container)) {
            throw "Required dataset folder is missing: $ClassPath"
        }
    }
}

$RunStamp = [DateTime]::UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'", [Globalization.CultureInfo]::InvariantCulture)
$ResearchRunRoot = Join-Path $ProjectRoot "runs\research"
$MultisignalRunRoot = Join-Path $ProjectRoot "runs\multisignal"
$ResearchModels = @("rgb_cnn", "resnet18_frozen", "resnet18_finetune", "hybrid_concat", "hybrid_attention")
$Ablations = @(
    @{ Name = "ablation-rgb"; Signals = @("rgb") },
    @{ Name = "ablation-rgb-edges"; Signals = @("rgb", "edges") },
    @{ Name = "ablation-rgb-texture"; Signals = @("rgb", "texture") },
    @{ Name = "ablation-rgb-edges-texture"; Signals = @("rgb", "edges", "texture") }
)

$PlannedRunIds = @(
    $ResearchModels | ForEach-Object { "suite-$_-seed$Seed-$RunStamp" }
    $Ablations | ForEach-Object { "$($_.Name)-seed$Seed-$RunStamp" }
)
foreach ($RunId in $PlannedRunIds) {
    $RunRoot = if ($RunId.StartsWith("suite-")) { $ResearchRunRoot } else { $MultisignalRunRoot }
    $RunDirectory = Join-Path $RunRoot $RunId
    if (Test-Path -LiteralPath $RunDirectory) {
        throw "Run folder already exists: $RunDirectory"
    }
}

$CacheDir = Join-Path $ProjectRoot "data\processed\streams_v1_224"
Write-Host "CUDA device:"
& $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the CUDA device name."
}

Write-Host "Preparing training and validation streams. The test split is not cached."
Invoke-ProjectPython -PythonArguments @(
    "-m", "src.banknote.preprocessing.cache_streams",
    "--dataset-dir", $DatasetDir,
    "--cache-dir", $CacheDir
)

function Invoke-ResearchModel {
    param(
        [Parameter(Mandatory)][string]$ModelName,
        [Parameter(Mandatory)][string]$RunId,
        [string[]]$AdditionalArguments = @()
    )

    $PythonArguments = @(
        "-m", "src.banknote.training.train_research_model",
        "--model", $ModelName,
        "--dataset-dir", $DatasetDir,
        "--cache-dir", $CacheDir,
        "--run-id", $RunId,
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--learning-rate", $LearningRate
    )
    $PythonArguments += $AdditionalArguments
    Write-Host "`nStarting $RunId"
    Invoke-ProjectPython -PythonArguments $PythonArguments
}

function Invoke-MultisignalAblation {
    param(
        [Parameter(Mandatory)][string]$ExperimentName,
        [Parameter(Mandatory)][string[]]$Signals
    )

    $RunId = "$ExperimentName-seed$Seed-$RunStamp"
    $PythonArguments = @(
        "-m", "src.banknote.training.train_multisignal",
        "--dataset-dir", $DatasetDir,
        "--cache-dir", $CacheDir,
        "--run-id", $RunId,
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--learning-rate", $LearningRate,
        "--signals"
    ) + $Signals
    Write-Host "`nStarting $RunId with signals: $($Signals -join ', ')"
    Invoke-ProjectPython -PythonArguments $PythonArguments
}

$RgbId = "suite-rgb_cnn-seed$Seed-$RunStamp"
$FrozenId = "suite-resnet18_frozen-seed$Seed-$RunStamp"
$FinetuneId = "suite-resnet18_finetune-seed$Seed-$RunStamp"
$ConcatId = "suite-hybrid_concat-seed$Seed-$RunStamp"
$AttentionId = "suite-hybrid_attention-seed$Seed-$RunStamp"

Invoke-ResearchModel -ModelName "rgb_cnn" -RunId $RgbId
Invoke-ResearchModel -ModelName "resnet18_frozen" -RunId $FrozenId
Invoke-ResearchModel -ModelName "resnet18_finetune" -RunId $FinetuneId -AdditionalArguments @(
    "--learning-rate", "0.0001",
    "--backbone-learning-rate", "0.00001",
    "--init-checkpoint", (Join-Path $ResearchRunRoot "$FrozenId\best_checkpoint.pth")
)
Invoke-ResearchModel -ModelName "hybrid_concat" -RunId $ConcatId
Invoke-ResearchModel -ModelName "hybrid_attention" -RunId $AttentionId

foreach ($Ablation in $Ablations) {
    Invoke-MultisignalAblation -ExperimentName $Ablation.Name -Signals $Ablation.Signals
}

Write-Host "`nTraining suite complete. No test evaluation was run."
Write-Host "Review validation results and select models before running the test evaluator."
