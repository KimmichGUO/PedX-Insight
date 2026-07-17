# Nightly PedX publish: aggregate PedX-Insight results -> copy to PedX-Visualizer -> push
# to the hosted DB. Registered in Windows Task Scheduler as "PedX Nightly Publish" (03:00
# daily) so the Globe stays current while the long batch run grinds through videos.
#
# Safe alongside a running batch: folders mid-analysis lack [A1]/[A2] and are skipped by
# the aggregation scripts; they are picked up the following night.

$ErrorActionPreference = "Continue"
$Python  = "C:\Users\markc\AppData\Local\Programs\Python\Python312\python.exe"
$Npm     = "C:\Users\markc\AppData\Local\nodejs\npm.cmd"
$Insight = "C:\Users\markc\Desktop\PedX-Insight"
$Viz     = "C:\Users\markc\Desktop\PedX-Visualizer"
$Log     = Join-Path $Insight "scripts\nightly_publish.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Add-Content -Path $Log -Value $line
}

Log "===== nightly publish start ====="
Set-Location $Insight

foreach ($script in @("get_all_pede_info.py", "get_all_video_info.py", "statistics_with_pdf_save.py")) {
    & $Python $script *>> $null
    if ($LASTEXITCODE -ne 0) { Log "FAIL $script (exit $LASTEXITCODE)"; Log "aborting"; exit 1 }
    Log "ok   $script"
}

Copy-Item -Path (Join-Path $Insight "summary_data\*.csv") -Destination (Join-Path $Viz "summary_data") -Force
Log "ok   copied summary_data CSVs"

Set-Location $Viz
& $Npm run db:publish *>> $null
if ($LASTEXITCODE -ne 0) { Log "FAIL npm run db:publish (exit $LASTEXITCODE)"; exit 1 }
Log "ok   db:publish (aggregate + coordinates + views + insights)"
Log "===== nightly publish done ====="
