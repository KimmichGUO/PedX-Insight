# Publish immutable configured revisions, with manifest validation and atomic activation.
$ErrorActionPreference = 'Stop'
$Insight = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Viz = if ($env:PEDX_VISUALIZER_ROOT) { $env:PEDX_VISUALIZER_ROOT } else { Join-Path (Split-Path $Insight -Parent) 'PedX-Visualizer' }
$Python = if ($env:PEDX_PYTHON) { $env:PEDX_PYTHON } else { 'C:\Users\markc\AppData\Local\Programs\Python\Python312\python.exe' }
$Node = if ($env:PEDX_NODE) { $env:PEDX_NODE } else { (Get-Command node -ErrorAction Stop).Source }
$Log = Join-Path $PSScriptRoot 'nightly_publish.log'
$ReleaseRef = if ($env:PEDX_PUBLISH_REF) { $env:PEDX_PUBLISH_REF } else { 'refs/heads/main' }
$VizRef = if ($env:PEDX_VISUALIZER_PUBLISH_REF) { $env:PEDX_VISUALIZER_PUBLISH_REF } else { 'refs/heads/main' }
$TempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$Work = Join-Path $TempRoot ('pedx_publish_' + [guid]::NewGuid().ToString('N'))
$ReleaseRoot = if ($env:PEDX_RELEASES_ROOT) { $env:PEDX_RELEASES_ROOT } else { Join-Path $env:LOCALAPPDATA 'PedX\releases' }
$Release = Join-Path $ReleaseRoot ([guid]::NewGuid().ToString('N'))

function Log($Message) { Add-Content -LiteralPath $Log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" }
function Snapshot($Repository, $Reference, $Destination, $Archive) {
    $Commit = (& git -C $Repository rev-parse --verify "$($Reference)^{commit}").Trim()
    if ($LASTEXITCODE -ne 0 -or $Commit -notmatch '^[a-f0-9]{40}$') { throw "Invalid publication reference: $Reference" }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & git -C $Repository archive --format=tar "--output=$Archive" $Commit
    if ($LASTEXITCODE -ne 0) { throw "Cannot archive $Repository" }
    & tar -xf $Archive -C $Destination
    if ($LASTEXITCODE -ne 0) { throw "Cannot extract $Repository" }
    return $Commit
}

$SavedEnvironment = @{}
foreach ($Name in @('PEDX_RESULTS_ROOT', 'PEDX_SUMMARY_ROOT', 'PEDX_IMMUTABLE_SOURCE_REVISION', 'PEDX_ENV_FILE', 'NODE_PATH')) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}
try {
    Log 'nightly publish start'
    New-Item -ItemType Directory -Path $Work -Force | Out-Null
    $InsightSnapshot = Join-Path $Work 'Insight'
    $VizSnapshot = Join-Path $Work 'Visualizer'
    $Commit = Snapshot $Insight $ReleaseRef $InsightSnapshot (Join-Path $Work 'insight.tar')
    $VizCommit = Snapshot $Viz $VizRef $VizSnapshot (Join-Path $Work 'visualizer.tar')
    if (-not (Test-Path -LiteralPath (Join-Path $InsightSnapshot 'scripts\build_release.py'))) { throw 'Configured Insight revision has no validated release builder' }
    if (-not (Test-Path -LiteralPath (Join-Path $VizSnapshot 'scripts\publish-release.js'))) { throw 'Configured Visualizer revision has no atomic publisher' }
    $env:PEDX_RESULTS_ROOT = Join-Path $Insight 'analysis_results'
    $env:PEDX_SUMMARY_ROOT = $Release
    $env:PEDX_IMMUTABLE_SOURCE_REVISION = $Commit
    $env:PEDX_ENV_FILE = if ($SavedEnvironment['PEDX_ENV_FILE']) { $SavedEnvironment['PEDX_ENV_FILE'] } elseif (Test-Path -LiteralPath (Join-Path $Viz '.env.local')) { Join-Path $Viz '.env.local' } else { Join-Path $Viz '.env' }
    $env:NODE_PATH = Join-Path $Viz 'node_modules'
    Push-Location $InsightSnapshot
    try {
        & $Python -B -m scripts.build_release --output $Release --results-root $env:PEDX_RESULTS_ROOT --mapping (Join-Path $Insight 'mapping.csv') *>> $Log
        if ($LASTEXITCODE -ne 0) { throw 'Release preparation failed; previous active release retained' }
    } finally { Pop-Location }
    Push-Location $VizSnapshot
    try {
        & $Node scripts/publish-release.js --csv-dir $Release *>> $Log
        if ($LASTEXITCODE -ne 0) { throw 'Atomic publication failed; previous active release retained' }
    } finally { Pop-Location }
    Log "published Insight=$Commit Visualizer=$VizCommit release=$Release"
    # Verify the absolute target before recursively removing this named scratch directory.
    $ResolvedWork = [System.IO.Path]::GetFullPath($Work)
    if (-not $ResolvedWork.StartsWith($TempRoot, [System.StringComparison]::OrdinalIgnoreCase) -or (Split-Path $ResolvedWork -Leaf) -notlike 'pedx_publish_*') { throw 'Refusing cleanup outside publication scratch directory' }
    Remove-Item -LiteralPath $ResolvedWork -Recurse -Force
} catch {
    Log "FAIL: $($_.Exception.Message); staging retained at $Work"
    exit 1
} finally {
    foreach ($Name in $SavedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($Name, $SavedEnvironment[$Name], 'Process') }
}
