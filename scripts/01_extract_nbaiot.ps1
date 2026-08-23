# ==========================================================
# N-BaIoT RAR Extraction Script
# Ham veriyi değiştirmeden data/interim alanına çıkarır.
# ==========================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$SourceRoot = Join-Path `
    $ProjectRoot `
    "data\raw\nbaiot\extracted"

$TargetRoot = Join-Path `
    $ProjectRoot `
    "data\interim\nbaiot_unpacked"

# ----------------------------------------------------------
# 7-Zip konumunu bul
# ----------------------------------------------------------

$SevenZipCandidates = @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe"
)

$SevenZip = $null

foreach ($candidate in $SevenZipCandidates) {

    if (Test-Path $candidate) {

        $SevenZip = $candidate
        break
    }
}

if (-not $SevenZip) {

    $command = Get-Command 7z `
        -ErrorAction SilentlyContinue

    if ($command) {

        $SevenZip = $command.Source
    }
}

if (-not $SevenZip) {

    throw "7-Zip bulunamadı. 7z.exe yolunu kontrol edin."
}

# ----------------------------------------------------------
# Kaynak klasörü doğrula
# ----------------------------------------------------------

if (-not (Test-Path $SourceRoot)) {

    throw "Kaynak veri klasörü bulunamadı: $SourceRoot"
}

New-Item `
    $TargetRoot `
    -ItemType Directory `
    -Force | Out-Null

Write-Host ""
Write-Host "N-BaIoT arşiv çıkarma işlemi başlatıldı."
Write-Host "7-Zip : $SevenZip"
Write-Host "Kaynak : $SourceRoot"
Write-Host "Hedef  : $TargetRoot"
Write-Host ""

$DeviceFolders = Get-ChildItem `
    $SourceRoot `
    -Directory

$ExtractedArchiveCount = 0
$CopiedBenignCount = 0

foreach ($deviceFolder in $DeviceFolders) {

    Write-Host "Cihaz işleniyor: $($deviceFolder.Name)"

    $TargetDeviceFolder = Join-Path `
        $TargetRoot `
        $deviceFolder.Name

    New-Item `
        $TargetDeviceFolder `
        -ItemType Directory `
        -Force | Out-Null

    # ------------------------------------------------------
    # Benign trafik dosyasını kopyala
    # ------------------------------------------------------

    $BenignFile = Join-Path `
        $deviceFolder.FullName `
        "benign_traffic.csv"

    if (Test-Path $BenignFile) {

        Copy-Item `
            $BenignFile `
            $TargetDeviceFolder `
            -Force

        $CopiedBenignCount++

        Write-Host "  Benign trafik kopyalandı."
    }
    else {

        Write-Warning `
            "Benign trafik bulunamadı: $($deviceFolder.Name)"
    }

    # ------------------------------------------------------
    # RAR saldırı arşivlerini çıkar
    # ------------------------------------------------------

    $RarFiles = Get-ChildItem `
        $deviceFolder.FullName `
        -Filter "*.rar" `
        -File

    foreach ($rarFile in $RarFiles) {

        $ArchiveName = `
            [System.IO.Path]::GetFileNameWithoutExtension(
                $rarFile.Name
            )

        $ArchiveTarget = Join-Path `
            $TargetDeviceFolder `
            $ArchiveName

        New-Item `
            $ArchiveTarget `
            -ItemType Directory `
            -Force | Out-Null

        Write-Host "  Çıkarılıyor: $($rarFile.Name)"

        & $SevenZip `
            x `
            $rarFile.FullName `
            "-o$ArchiveTarget" `
            -y | Out-Null

        if ($LASTEXITCODE -ne 0) {

            throw `
                "Arşiv çıkarma başarısız: $($rarFile.FullName)"
        }

        $ExtractedArchiveCount++
    }

    Write-Host ""
}

# ----------------------------------------------------------
# Özet
# ----------------------------------------------------------

Write-Host "============================================================"
Write-Host "N-BaIoT çıkarma işlemi tamamlandı."
Write-Host "Cihaz sayısı              : $($DeviceFolders.Count)"
Write-Host "Kopyalanan benign dosyası : $CopiedBenignCount"
Write-Host "Çıkarılan RAR arşivi      : $ExtractedArchiveCount"
Write-Host "Hedef klasör              : $TargetRoot"
Write-Host "============================================================"