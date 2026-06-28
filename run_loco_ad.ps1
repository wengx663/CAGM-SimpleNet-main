param(
    [string]$DataPath = "D:\datasets\loco_ad",
    [int]$Gpu = 0,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [int]$MetaEpochs = 5,
    [int]$GanEpochs = 4,
    [string]$ResultsPath = "results",
    [string]$RunName = "se_gnnk5_m5",
    [string]$PythonExe = "D:\anaconda\envs\yolo11\python.exe",
    [switch]$DryRun,
    [string]$FeatureAttention = "se",
    [int]$AttentionReduction = 16,
    [int]$AttentionKernelSize = 3,
    [string]$FeatureL2Norm = "false",
    [int]$ScoreTopK = 0,
    [int]$MixNoise = 2,
    [string]$CosLr = "true",
    [string]$GlobalBranch = "false",
    [double]$GlobalLossWeight = 0.2,
    [double]$GlobalScoreWeight = 0.3,
    [double]$GlobalNoiseStd = 0,
    [string]$GlobalNNBranch = "true",
    [double]$GlobalNNWeight = 0.2,
    [int]$GlobalNNK = 5,
    [switch]$UseLayer4,
    [switch]$Resume,
    [string[]]$Datasets = @(
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors"
    )
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Convert-ToBool {
    param([object]$Value)

    if ($Value -is [bool]) {
        return $Value
    }

    $text = "$Value".Trim().ToLowerInvariant()
    return @("1", "true", "yes", "y", "on") -contains $text
}

$FeatureL2NormEnabled = Convert-ToBool $FeatureL2Norm
$CosLrEnabled = Convert-ToBool $CosLr
$GlobalBranchEnabled = Convert-ToBool $GlobalBranch
$GlobalNNBranchEnabled = Convert-ToBool $GlobalNNBranch

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$arguments = @(
    "main.py",
    "--gpu", "$Gpu",
    "--seed", "0",
    "--log_group", "simplenet_loco_ad",
    "--log_project", "LOCO_AD_Results",
    "--results_path", $ResultsPath,
    "--run_name", $RunName,
    "net",
    "-b", "wideresnet50",
    "-le", "layer2",
    "-le", "layer3",
    "--pretrain_embed_dimension", "1536",
    "--target_embed_dimension", "1536",
    "--patchsize", "3",
    "--meta_epochs", "$MetaEpochs",
    "--embedding_size", "256",
    "--gan_epochs", "$GanEpochs",
    "--noise_std", "0.015",
    "--dsc_hidden", "1024",
    "--dsc_layers", "2",
    "--dsc_margin", ".5",
    "--pre_proj", "1",
    "--mix_noise", "$MixNoise",
    "--feature_attention", $FeatureAttention,
    "--attention_reduction", "$AttentionReduction",
    "--attention_kernel_size", "$AttentionKernelSize",
    "--score_top_k", "$ScoreTopK"
)

if ($FeatureL2NormEnabled) {
    $arguments += "--feature_l2_norm"
}

if ($CosLrEnabled) {
    $arguments += "--cos_lr"
}

if ($GlobalBranchEnabled) {
    $arguments += @(
        "--global_branch",
        "--global_loss_weight", "$GlobalLossWeight",
        "--global_score_weight", "$GlobalScoreWeight",
        "--global_noise_std", "$GlobalNoiseStd"
    )
}

if ($GlobalNNBranchEnabled) {
    $arguments += @(
        "--global_nn_branch",
        "--global_nn_weight", "$GlobalNNWeight",
        "--global_nn_k", "$GlobalNNK"
    )
}

if ($UseLayer4) {
    $arguments += @("-le", "layer4")
}

if ($Resume) {
    $arguments += "--resume"
}

$arguments += @(
    "dataset",
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--resize", "329",
    "--imagesize", "288"
)

foreach ($dataset in $Datasets) {
    $arguments += @("-d", $dataset)
}

$arguments += @("mvtec", $DataPath)

Write-Host "Running with Python: $PythonExe"
Write-Host "MetaEpochs=$MetaEpochs GanEpochs=$GanEpochs Datasets=$($Datasets -join ',')"
Write-Host "Enhance: attention=$FeatureAttention l2_norm=$FeatureL2NormEnabled score_top_k=$ScoreTopK mix_noise=$MixNoise cos_lr=$CosLrEnabled use_layer4=$UseLayer4 resume=$Resume"
Write-Host "Global branch: enabled=$GlobalBranchEnabled loss_weight=$GlobalLossWeight score_weight=$GlobalScoreWeight noise_std=$GlobalNoiseStd"
Write-Host "Global NN branch: enabled=$GlobalNNBranchEnabled weight=$GlobalNNWeight k=$GlobalNNK calibrated=true"

if ($DryRun) {
    Write-Host "Dry run command:"
    Write-Host "`"$PythonExe`" -u $($arguments -join ' ')"
    exit 0
}

& $PythonExe -u @arguments
