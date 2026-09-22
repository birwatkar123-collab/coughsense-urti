$ErrorActionPreference = 'Stop'
$REVIEW = Join-Path $PSScriptRoot 'urti-audio-reviews.json'
$PY = 'python'

if (-not (Test-Path -LiteralPath $REVIEW)) {
    Write-Host "Review file not found: $REVIEW"
    Write-Host ''
    Write-Host '1. Open D:\urti\experiments\listening-review\index.html in a browser'
    Write-Host '2. Complete all 90 recordings manually'
    Write-Host '3. Click Download reviews'
    Write-Host "4. Save/copy the exported file to: $REVIEW"
    Write-Host '5. Rerun this script, or run:'
    Write-Host "   python analyze_reviews.py --review `"$REVIEW`""
    exit 2
}

& $PY (Join-Path $PSScriptRoot 'analyze_reviews.py') '--review' $REVIEW
exit $LASTEXITCODE