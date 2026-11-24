$ErrorActionPreference = 'Stop'
$reportPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\output\reports\Class_Adaptive_Augmentation_Project_Proposal_and_Related_Work_2026-11-03.docx')).Path
$reportPdf = Join-Path $PSScriptRoot 'render\report.pdf'
$reportWord = $null
$reportDoc = $null
try {
    $reportWord = New-Object -ComObject Word.Application
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDoc = $reportWord.Documents.Open($reportPath, $false, $true)
    $reportDoc.Repaginate()
    $reportDoc.ExportAsFixedFormat($reportPdf, 17)
    Write-Output ('Rendered pages: ' + $reportDoc.ComputeStatistics(2))
    Write-Output $reportPdf
} finally {
    if ($null -ne $reportDoc) {
        try { $reportDoc.Close(0) } catch { Write-Output ('Word document cleanup: ' + $_.Exception.Message) }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportDoc)
    }
    if ($null -ne $reportWord) {
        try { $reportWord.Quit() } catch { Write-Output ('Word application cleanup: ' + $_.Exception.Message) }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportWord)
    }
}
