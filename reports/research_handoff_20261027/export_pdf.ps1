$ErrorActionPreference = 'Stop'
$handoffRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$handoffDocx = Join-Path $handoffRoot 'output\reports\RDD_Research_Handoff_2026-10-27.docx'
$handoffPdf = Join-Path $PSScriptRoot 'handoff_qa.pdf'
$handoffOffice = New-Object -ComObject Word.Application
$handoffOffice.Visible = $false
$handoffOffice.DisplayAlerts = 0
$handoffDocument = $null
try {
    $handoffDocument = $handoffOffice.Documents.Open($handoffDocx, $false, $true)
    $handoffDocument.Repaginate()
    $handoffDocument.ExportAsFixedFormat($handoffPdf, 17)
    if (-not (Test-Path -LiteralPath $handoffPdf)) { throw 'PDF export did not create the expected file.' }
    Write-Output "QA PDF exported: $handoffPdf"
} finally {
    if ($null -ne $handoffDocument) {
        try { $handoffDocument.Close(0) } catch { Write-Output 'The export document was already closed.' }
    }
    try { $handoffOffice.Quit() } catch { Write-Output 'The export application already disconnected.' }
}
