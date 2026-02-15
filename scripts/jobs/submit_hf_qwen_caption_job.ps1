param(
  [string]$CodeRepo = "YOUR_USERNAME/ace-step-lora-studio",
  [string]$DatasetRepo = "",
  [string]$DatasetRevision = "main",
  [string]$DatasetSubdir = "",
  [string]$Backend = "local",
  [string]$ModelId = "Qwen/Qwen2-Audio-7B-Instruct",
  [string]$EndpointUrl = "",
  [string]$Device = "auto",
  [string]$TorchDtype = "auto",
  [string]$Prompt = "",
  [double]$SegmentSeconds = 30.0,
  [double]$OverlapSeconds = 2.0,
  [int]$MaxNewTokens = 384,
  [double]$Temperature = 0.1,
  [string]$OutputDir = "/workspace/qwen_annotations",
  [string]$UploadRepo = "",
  [switch]$UploadPrivate,
  [switch]$CopyAudio,
  [switch]$KeepRawOutputs,
  [switch]$WriteInplaceSidecars,
  [string]$Flavor = "a10g-large",
  [string]$Timeout = "8h",
  [switch]$Detach
)

$ErrorActionPreference = "Stop"

if (-not $DatasetRepo) {
  throw "Provide -DatasetRepo (HF dataset repo containing audio files)."
}

if ($Backend -eq "hf_endpoint" -and -not $EndpointUrl) {
  throw "Backend hf_endpoint requires -EndpointUrl."
}

$secretArgs = @("--secrets", "HF_TOKEN")

$datasetSubdirArgs = ""
if ($DatasetSubdir) {
  $datasetSubdirArgs = "--dataset-subdir `"$DatasetSubdir`""
}

$endpointArgs = ""
if ($EndpointUrl) {
  $endpointArgs = "--endpoint-url `"$EndpointUrl`""
}

$uploadArgs = ""
if ($UploadRepo) {
  $uploadArgs = "--upload-repo `"$UploadRepo`""
  if ($UploadPrivate.IsPresent) {
    $uploadArgs += " --upload-private"
  }
}

$copyAudioArg = ""
if ($CopyAudio.IsPresent) {
  $copyAudioArg = "--copy-audio"
}

$keepRawArg = ""
if ($KeepRawOutputs.IsPresent) {
  $keepRawArg = "--keep-raw-outputs"
}

$writeInplaceArg = ""
if ($WriteInplaceSidecars.IsPresent) {
  $writeInplaceArg = "--write-inplace-sidecars"
}

$promptArg = ""
if ($Prompt) {
  $escapedPrompt = $Prompt.Replace('"', '\"')
  $promptArg = "--prompt `"$escapedPrompt`""
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
python scripts/annotations/qwen_caption_dataset.py \
  --dataset-repo "$DatasetRepo" \
  --dataset-revision "$DatasetRevision" \
  $datasetSubdirArgs \
  --backend "$Backend" \
  --model-id "$ModelId" \
  $endpointArgs \
  --device "$Device" \
  --torch-dtype "$TorchDtype" \
  --segment-seconds $SegmentSeconds \
  --overlap-seconds $OverlapSeconds \
  --max-new-tokens $MaxNewTokens \
  --temperature $Temperature \
  --output-dir "$OutputDir" \
  $promptArg \
  $copyAudioArg \
  $keepRawArg \
  $writeInplaceArg \
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

Write-Host "Submitting Qwen caption HF Job with flavor=$Flavor timeout=$Timeout ..."
Write-Host "Dataset repo: $DatasetRepo"
Write-Host "Code repo: $CodeRepo"
if ($UploadRepo) {
  Write-Host "Will upload exported annotations to: $UploadRepo"
}

& hf @argsList

