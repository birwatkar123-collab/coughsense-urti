param([string]$ZipPath = 'D:/ECGVisionAI/public_dataset_v3.zip')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $entry = @($zip.Entries | Where-Object { $_.Name -eq 'metadata_compiled.csv' })
    if ($entry.Count -ne 1) { throw 'Expected one metadata_compiled.csv' }
    $reader = [IO.StreamReader]::new($entry[0].Open())
    try { $rows = @($reader.ReadToEnd() | ConvertFrom-Csv -WarningAction SilentlyContinue) }
    finally { $reader.Dispose() }
    $audio = @{}
    foreach ($e in $zip.Entries) {
        if ([IO.Path]::GetExtension($e.Name) -in @('.wav','.ogg','.webm')) {
            $id = [IO.Path]::GetFileNameWithoutExtension($e.Name)
            if (-not $audio.ContainsKey($id)) { $audio[$id] = 0 }
            $audio[$id]++
        }
    }
    $stats = [ordered]@{
        zip_path = $ZipPath
        metadata_rows = $rows.Count
        unique_uuids = @($rows.uuid | Sort-Object -Unique).Count
        audio_files = ($audio.Values | Measure-Object -Sum).Sum
        duplicate_audio_ids = @($audio.Values | Where-Object { $_ -gt 1 }).Count
        rows_missing_audio = 0
        no_expert_diagnosis = 0
        any_expert_diagnosis = 0
        single_expert_diagnosis = 0
        multiple_expert_diagnoses = 0
        binary_disagreement = 0
        exact_diagnosis_disagreement = 0
        consistent_upper_infection = 0
        consistent_non_upper_infection = 0
        majority_upper_infection = 0
        majority_non_upper_infection = 0
        binary_tie = 0
        labeled_any_poor_or_no_cough_quality = 0
        diagnosis_votes = @{}
    }
    $details = foreach ($row in $rows) {
        if (-not $audio.ContainsKey($row.uuid)) { $stats.rows_missing_audio++ }
        $votes = @(foreach ($i in 1..4) {
            $v = $row."diagnosis_$i"
            if (-not [string]::IsNullOrWhiteSpace($v)) { $v }
        })
        if ($votes.Count -eq 0) { $stats.no_expert_diagnosis++; continue }
        $stats.any_expert_diagnosis++
        if ($votes.Count -eq 1) { $stats.single_expert_diagnosis++ }
        else { $stats.multiple_expert_diagnoses++ }
        foreach ($v in $votes) {
            if (-not $stats.diagnosis_votes.ContainsKey($v)) { $stats.diagnosis_votes[$v] = 0 }
            $stats.diagnosis_votes[$v]++
        }
        $positive = @($votes | Where-Object { $_ -eq 'upper_infection' }).Count
        $negative = $votes.Count - $positive
        if (@($votes | Sort-Object -Unique).Count -gt 1) { $stats.exact_diagnosis_disagreement++ }
        if ($positive -gt 0 -and $negative -gt 0) { $stats.binary_disagreement++ }
        elseif ($positive -gt 0) { $stats.consistent_upper_infection++ }
        else { $stats.consistent_non_upper_infection++ }
        if ($positive -gt $negative) { $stats.majority_upper_infection++ }
        elseif ($negative -gt $positive) { $stats.majority_non_upper_infection++ }
        else { $stats.binary_tie++ }
        $qualityFlag = @(foreach ($i in 1..4) {
            if ($row."quality_$i" -in @('poor','no_cough')) { $i }
        }).Count -gt 0
        if ($qualityFlag) { $stats.labeled_any_poor_or_no_cough_quality++ }
        [pscustomobject]@{
            uuid = $row.uuid
            expert_votes = $votes.Count
            upper_votes = $positive
            other_votes = $negative
            binary_conflict = ($positive -gt 0 -and $negative -gt 0)
            any_poor_or_no_cough_quality = $qualityFlag
            audio_present = $audio.ContainsKey($row.uuid)
        }
    }
    if ($stats.no_expert_diagnosis + $stats.any_expert_diagnosis -ne $rows.Count) { throw 'Row count mismatch' }
    if ($stats.consistent_upper_infection + $stats.consistent_non_upper_infection + $stats.binary_disagreement -ne $stats.any_expert_diagnosis) { throw 'Vote count mismatch' }
    $stats | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $PSScriptRoot 'dataset_inspection.json') -Encoding utf8
    $details | Export-Csv (Join-Path $PSScriptRoot 'expert_label_audit.csv') -NoTypeInformation -Encoding utf8
    $stats | ConvertTo-Json -Depth 4
} finally { $zip.Dispose() }
