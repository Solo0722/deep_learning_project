$ErrorActionPreference = 'Stop'
$reportRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$reportDocx = Join-Path $reportRoot 'output\reports\RDD_Experiment_Report_2026-10-27.docx'
$reportPdf = Join-Path $reportRoot 'output\reports\RDD_Experiment_Report_2026-10-27.pdf'
$reportOffice = New-Object -ComObject Word.Application
$reportOffice.Visible = $false
$reportOffice.DisplayAlerts = 0
$reportDocument = $null
try {
    $reportDocument = $reportOffice.Documents.Open($reportDocx, $false, $true)
    $reportDocument.Repaginate()
    $reportDocument.ExportAsFixedFormat($reportPdf, 17)
    if (-not (Test-Path -LiteralPath $reportPdf)) { throw 'PDF export did not create the expected file.' }
    Write-Output "PDF exported: $reportPdf"
} finally {
    if ($null -ne $reportDocument) {
        try { $reportDocument.Close(0) } catch { Write-Output 'The native export document was already closed.' }
    }
    try { $reportOffice.Quit() } catch { Write-Output 'The native export application has already disconnected.' }
}
