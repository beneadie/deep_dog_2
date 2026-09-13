param(
    [string]$InputFile = "deepdog2_demo.mp4",
    [string]$OutputFile = "deepdog2_demo_web_hd.mp4"
)

$ErrorActionPreference = "Stop"

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Error "FFmpeg was not found. Install FFmpeg, restart PowerShell, and run this script again."
    exit 1
}

if (-not (Test-Path -LiteralPath $InputFile)) {
    Write-Error "Input video not found: $InputFile"
    exit 1
}

if ((Resolve-Path -LiteralPath $InputFile).Path -eq (Join-Path (Get-Location) $OutputFile)) {
    Write-Error "The output file must be different from the original video."
    exit 1
}

Write-Host "Compressing $InputFile to $OutputFile ..."

& $ffmpeg.Source `
    -y `
    -i $InputFile `
    -vf "scale=-2:720" `
    -c:v libx264 `
    -preset medium `
    -crf 27 `
    -c:a aac `
    -b:a 96k `
    -movflags +faststart `
    $OutputFile

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$source = Get-Item -LiteralPath $InputFile
$result = Get-Item -LiteralPath $OutputFile
Write-Host "Done."
Write-Host ("Original: {0:N1} MB" -f ($source.Length / 1MB))
Write-Host ("Compressed: {0:N1} MB" -f ($result.Length / 1MB))
