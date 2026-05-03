# Setup script — fixes corporate SSL inspection issue.
# Lam's network does SSL/TLS inspection with a self-signed root CA.
# Python and curl_cffi don't trust it by default. This script:
#   1. Exports all Windows root certs into a single PEM bundle
#   2. Sets the env vars so yfinance / requests / curl_cffi use it
#
# Run this ONCE per PowerShell session before running quick_analysis.py:
#     . .\setup_corp_ssl.ps1
#
# Or to make permanent for your user, add the bundle path to user env vars.

$bundlePath = Join-Path $PSScriptRoot "corp-ca-bundle.pem"

if (-not (Test-Path $bundlePath)) {
    Write-Host "Exporting Windows root CAs to $bundlePath ..." -ForegroundColor Cyan
    $certs = @()
    $certs += Get-ChildItem Cert:\LocalMachine\Root -ErrorAction SilentlyContinue
    $certs += Get-ChildItem Cert:\CurrentUser\Root  -ErrorAction SilentlyContinue
    $certs += Get-ChildItem Cert:\LocalMachine\CA   -ErrorAction SilentlyContinue
    $certs += Get-ChildItem Cert:\CurrentUser\CA    -ErrorAction SilentlyContinue

    $sb = New-Object System.Text.StringBuilder
    foreach ($c in $certs) {
        if ($null -eq $c.RawData) { continue }
        [void]$sb.AppendLine("# Subject: $($c.Subject)")
        [void]$sb.AppendLine("# Issuer:  $($c.Issuer)")
        [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
        $b64 = [System.Convert]::ToBase64String($c.RawData)
        for ($i = 0; $i -lt $b64.Length; $i += 64) {
            $len = [Math]::Min(64, $b64.Length - $i)
            [void]$sb.AppendLine($b64.Substring($i, $len))
        }
        [void]$sb.AppendLine("-----END CERTIFICATE-----")
        [void]$sb.AppendLine("")
    }
    [System.IO.File]::WriteAllText($bundlePath, $sb.ToString(), [System.Text.Encoding]::ASCII)
    Write-Host "Wrote $($certs.Count) certs." -ForegroundColor Green
} else {
    Write-Host "Using existing bundle: $bundlePath" -ForegroundColor Gray
}

$env:CURL_CA_BUNDLE     = $bundlePath
$env:REQUESTS_CA_BUNDLE = $bundlePath
$env:SSL_CERT_FILE      = $bundlePath

Write-Host "✅ SSL env vars set for this session." -ForegroundColor Green
Write-Host "   CURL_CA_BUNDLE     = $bundlePath"
Write-Host "   REQUESTS_CA_BUNDLE = $bundlePath"
Write-Host "   SSL_CERT_FILE      = $bundlePath"
