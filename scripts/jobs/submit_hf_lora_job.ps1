param(
  [string]$CodeRepo = "YOUR_USERNAME/ace-step-lora-studio",
  [string]$DatasetRepo = "",
  [string]$DatasetRevision = "main",
  [string]$DatasetSubdir = "",
  [string]$ModelConfig = "acestep-v15-base",
  [string]$Flavor = "a10g-large",
  [string]$Timeout = "8h",
  [int]$Epochs = 20,
  [int]$BatchSize = 1,
  [int]$GradAccum = 1,
  [string]$OutputDir = "/workspace/output",
  [string]$UploadRepo = "",
  [switch]$UploadPrivate,
  [switch]$Detach
)

$ErrorActionPreference = "Stop"

if (-not $DatasetRepo) {
  throw "Provide -DatasetRepo (HF dataset repo containing your audio + optional sidecars)."
}

$secretArgs = @("--secrets", "HF_TOKEN")

$uploadArgs = ""
if ($UploadRepo) {
  $uploadArgs = "--upload-repo `"$UploadRepo`""
  if ($UploadPrivate.IsPresent) {
    $uploadArgs += " --upload-private"
  }
}

$datasetSubdirArgs = ""
if ($DatasetSubdir) {
  $datasetSubdirArgs = "--dataset-subdir `"$DatasetSubdir`""
}

$detachArg = ""
if ($Detach.IsPresent) {
  $detachArg = "--detach"
}

$jobCommand = @"
set -e
python -m pip install --no-cache-dir --upgrade pip
git clone https://huggingface.co/$CodeRepo /workspace/code
cd /workspace/code
python -m pip install --no-cache-dir -r requirements.txt
python lora_train.py \
  --dataset-repo "$DatasetRepo" \
  --dataset-revision "$DatasetRevision" \
  $datasetSubdirArgs \
  --model-config "$ModelConfig" \
  --device auto \
  --num-epochs $Epochs \
  --batch-size $BatchSize \
  --grad-accum $GradAccum \
  --output-dir "$OutputDir" \
  $uploadArgs
"@

$argsList = @(
  "jobs", "run",
  "--flavor", $Flavor,
  "--timeout", $Timeout
) + $secretArgs

if ($detachArg) {
  $argsList += $detachArg
}

$argsList += @(
  "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
  "bash", "-lc", $jobCommand
)

Write-Host "Submitting HF Job with flavor=$Flavor timeout=$Timeout ..."
Write-Host "Dataset repo: $DatasetRepo"
Write-Host "Code repo: $CodeRepo"
if ($UploadRepo) {
  Write-Host "Will upload final adapter to: $UploadRepo"
}

& hf @argsList
