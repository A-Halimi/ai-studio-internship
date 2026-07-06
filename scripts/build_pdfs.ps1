# Convert every handout in handouts\html\ to a PDF in handouts\pdf\
# using headless Chrome (or Edge as fallback).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_pdfs.ps1
#
# Uses Start-Process so Chrome's stderr chatter can't trip PowerShell 5.1's
# NativeCommandError handling.

$root = Split-Path -Parent $PSScriptRoot

$browser = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $browser) {
    Write-Error "Neither Chrome nor Edge was found - cannot render PDFs."
    exit 1
}
Write-Host "Rendering with: $browser"

$pdfDir = Join-Path $root "handouts\pdf"
New-Item -ItemType Directory -Force -Path $pdfDir | Out-Null

$failed = 0
Get-ChildItem (Join-Path $root "handouts\html\*.html") | Sort-Object Name | ForEach-Object {
    $pdf = Join-Path $pdfDir ($_.BaseName + ".pdf")
    $uri = "file:///" + ($_.FullName -replace '\\', '/')
    $args = @(
        "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--print-to-pdf=`"$pdf`"", "`"$uri`""
    )
    $p = Start-Process -FilePath $browser -ArgumentList $args -Wait -PassThru -WindowStyle Hidden
    if ((Test-Path $pdf) -and ((Get-Item $pdf).Length -gt 10KB)) {
        Write-Host ("  OK    {0,-28} {1,8:n0} KB" -f $_.BaseName, ((Get-Item $pdf).Length / 1KB))
    } else {
        Write-Host ("  FAIL  {0}  (exit {1})" -f $_.BaseName, $p.ExitCode)
        $script:failed++
    }
}

if ($failed -gt 0) { Write-Host "$failed handout(s) failed."; exit 1 }
Write-Host "All handout PDFs generated in handouts\pdf\"
exit 0
