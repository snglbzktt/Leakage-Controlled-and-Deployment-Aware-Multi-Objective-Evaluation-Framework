param(
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$ProjectName = Split-Path $ProjectRoot -Leaf
$ParentDir   = Split-Path $ProjectRoot -Parent
$BackupDir  = Join-Path $ParentDir "${ProjectName}_V1_ARCHIVE"

Write-Host ""
Write-Host "Proje kökü : $ProjectRoot" -ForegroundColor Cyan
Write-Host "V1 arşivi  : $BackupDir" -ForegroundColor Cyan

# Proje kökü doğrulaması
$RequiredFolders = @(
    "src",
    "scripts",
    "configs",
    "results"
)

foreach ($Folder in $RequiredFolders) {
    $FullPath = Join-Path $ProjectRoot $Folder

    if (-not (Test-Path $FullPath)) {
        throw "Proje kökünde değilsin. Eksik klasör: $Folder"
    }
}

# ----------------------------------------------------------
# 1. V1 arşivini oluştur
# ----------------------------------------------------------
if (-not $SkipBackup) {

    Write-Host ""
    Write-Host "V1 arşivi oluşturuluyor..." -ForegroundColor Yellow

    New-Item `
        -ItemType Directory `
        -Path $BackupDir `
        -Force |
        Out-Null

    $RoboArgs = @(
        $ProjectRoot
        $BackupDir
        "/E"
        "/R:1"
        "/W:1"
        "/XD"
        ".venv"
        "venv"
        "data"
        ".git"
        "__pycache__"
        ".pytest_cache"
        ".ruff_cache"
        "results\v2"
        "/XF"
        "*.pyc"
        "*.pyo"
        "*.tmp"
        "*.temp"
        "*.bak"
        "*.zip"
    )

    & robocopy @RoboArgs

    $RoboCode = $LASTEXITCODE

    # Robocopy 0-7 arası başarılı kabul edilir.
    if ($RoboCode -gt 7) {
        throw "V1 arşivleme başarısız. Robocopy kodu: $RoboCode"
    }

    Write-Host "V1 arşivi başarıyla oluşturuldu." -ForegroundColor Green
}
else {
    Write-Host "V1 yedekleme atlandı." -ForegroundColor Yellow
}

# ----------------------------------------------------------
# 2. Q1 V2 klasörlerini oluştur
# ----------------------------------------------------------
$V2Folders = @(
    "docs\v2",
    "docs\v2\licenses",
    "configs\protocols\v2",
    "configs\experiments\v2",
    "results\v2\audit",
    "results\v2\early_lodo",
    "results\v2\vs2_ablation",
    "results\v2\tabular_baselines",
    "results\v2\fair_training_controls",
    "results\v2\ptq_calibration",
    "results\v2\cnn_order_ablation",
    "results\v2\device_holdout",
    "results\v2\second_dataset",
    "results\v2\runtime_profiles",
    "results\v2\statistics",
    "results\v2\publication",
    "results\v2\manifests"
)

foreach ($RelativePath in $V2Folders) {
    New-Item `
        -ItemType Directory `
        -Path (Join-Path $ProjectRoot $RelativePath) `
        -Force |
        Out-Null
}

Write-Host "V2 klasörleri oluşturuldu." -ForegroundColor Green

# ----------------------------------------------------------
# 3. V1 dondurma notu
# ----------------------------------------------------------
$FreezeTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$FreezeText = @"
# V1 Freeze Note

Freeze time: $FreezeTime

Original project:
$ProjectRoot

V1 archive:
$BackupDir

Rules:
- V1 results must not be overwritten.
- New experiments must write only under results/v2.
- Existing 150 primary experiment outputs are preserved.
- Existing VS2 split artifacts are preserved.
- Raw data and virtual environment were not copied.
"@

$FreezeFile = Join-Path `
    $ProjectRoot `
    "docs\v2\V1_FREEZE_NOTE.md"

Set-Content `
    -Path $FreezeFile `
    -Value $FreezeText `
    -Encoding UTF8

# ----------------------------------------------------------
# 4. Veri lisansı kayıt şablonu
# ----------------------------------------------------------
$LicenseRegister = Join-Path `
    $ProjectRoot `
    "docs\v2\licenses\DATASET_LICENSE_REGISTER.csv"

$LicenseRows = @(
    [PSCustomObject]@{
        dataset_name              = "N-BaIoT"
        canonical_source          = ""
        access_date               = ""
        license_or_usage_terms    = ""
        academic_use              = ""
        commercial_use            = ""
        redistribution            = ""
        derivative_data_sharing   = ""
        citation_required         = ""
        required_citation         = ""
        archive_sha256            = ""
        evidence_file             = ""
        notes                     = ""
    },
    [PSCustomObject]@{
        dataset_name              = "TON_IoT"
        canonical_source          = ""
        access_date               = ""
        license_or_usage_terms    = ""
        academic_use              = ""
        commercial_use            = ""
        redistribution            = ""
        derivative_data_sharing   = ""
        citation_required         = ""
        required_citation         = ""
        archive_sha256            = ""
        evidence_file             = ""
        notes                     = ""
    },
    [PSCustomObject]@{
        dataset_name              = "Edge-IIoTset"
        canonical_source          = ""
        access_date               = ""
        license_or_usage_terms    = ""
        academic_use              = ""
        commercial_use            = ""
        redistribution            = ""
        derivative_data_sharing   = ""
        citation_required         = ""
        required_citation         = ""
        archive_sha256            = ""
        evidence_file             = ""
        notes                     = ""
    },
    [PSCustomObject]@{
        dataset_name              = "CICIoT2023"
        canonical_source          = ""
        access_date               = ""
        license_or_usage_terms    = ""
        academic_use              = ""
        commercial_use            = ""
        redistribution            = ""
        derivative_data_sharing   = ""
        citation_required         = ""
        required_citation         = ""
        archive_sha256            = ""
        evidence_file             = ""
        notes                     = ""
    }
)

$LicenseRows |
    Export-Csv `
        -Path $LicenseRegister `
        -NoTypeInformation `
        -Encoding UTF8

# ----------------------------------------------------------
# 5. Kod ve rapor SHA-256 manifesti
# ----------------------------------------------------------
$ManifestRoots = @(
    "src",
    "scripts",
    "configs",
    "research",
    "docs",
    "results\reports",
    "results\publication",
    "results\vs2",
    "models\preprocessing"
)

$ManifestRows = foreach ($RelativeRoot in $ManifestRoots) {

    $AbsoluteRoot = Join-Path $ProjectRoot $RelativeRoot

    if (Test-Path $AbsoluteRoot) {

        Get-ChildItem `
            -Path $AbsoluteRoot `
            -Recurse `
            -File |
        Where-Object {
            $_.Extension -notin @(
                ".pyc",
                ".pyo",
                ".tmp",
                ".bak"
            )
        } |
        ForEach-Object {

            $Hash = Get-FileHash `
                -Path $_.FullName `
                -Algorithm SHA256

            [PSCustomObject]@{
                relative_path = $_.FullName.Substring(
                    $ProjectRoot.Length + 1
                )
                size_bytes = $_.Length
                sha256 = $Hash.Hash.ToLowerInvariant()
            }
        }
    }
}

$ManifestFile = Join-Path `
    $ProjectRoot `
    "results\v2\manifests\v1_code_and_results_sha256.csv"

$ManifestRows |
    Sort-Object relative_path |
    Export-Csv `
        -Path $ManifestFile `
        -NoTypeInformation `
        -Encoding UTF8

# ----------------------------------------------------------
# 6. Python sözdizimi kontrolü
# ----------------------------------------------------------
$PythonExe = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe --version

& $PythonExe `
    -m compileall `
    -q `
    src `
    scripts

if ($LASTEXITCODE -ne 0) {
    throw "Python sözdizimi kontrolü başarısız."
}

# ----------------------------------------------------------
# 7. Sonuç özeti
# ----------------------------------------------------------
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "FAZ 0 BAŞLANGIÇ HAZIRLIĞI TAMAMLANDI" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "V1 arşivi      : $BackupDir"
Write-Host "Dondurma notu  : $FreezeFile"
Write-Host "Hash manifesti : $ManifestFile"
Write-Host "Lisans kaydı   : $LicenseRegister"
Write-Host "V2 sonuç yolu  : $(Join-Path $ProjectRoot 'results\v2')"
Write-Host "==================================================" -ForegroundColor Cyan
