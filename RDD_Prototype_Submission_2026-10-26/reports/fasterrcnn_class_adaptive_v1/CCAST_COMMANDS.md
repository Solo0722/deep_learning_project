# Stage 5: upload, submit, inspect

Prepared 2026-10-26. Run each block in the named terminal. Stop if a block reports
an error. The upload is small (93,081 bytes). Submission runs from a CCAST login
node; the PBS job requests the GPU and activates the environment automatically.

## 1. Windows PowerShell: verify and upload

```powershell
& {
    $ErrorActionPreference = "Stop"
    $classBundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\fasterrcnn_class_adaptive_v1.tar.gz"
    $classExpected = "3e4abe51cbee1fb3d609fd3626e9ea96bbc402aa61c0e59dfe692ac6df48b661"

    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $classBundle).Hash -ne $classExpected) {
        throw "Package checksum mismatch."
    }

    scp $classBundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
    Write-Output "Upload completed."
}
```

If disconnected, reconnect to CCAST with `ssh solomon.owusu.1@prime.ccast.ndsu.edu`.

## 2. CCAST login node: verify, extract and submit

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research

    printf '%s  %s\n' \
      '3e4abe51cbee1fb3d609fd3626e9ea96bbc402aa61c0e59dfe692ac6df48b661' \
      'fasterrcnn_class_adaptive_v1.tar.gz' | sha256sum -c -

    if [ -e fasterrcnn_class_adaptive_v1 ]; then
        printf 'Package directory already exists; stop and inspect it.\n' >&2
        exit 1
    fi

    tar --keep-old-files -xzf fasterrcnn_class_adaptive_v1.tar.gz
    cd fasterrcnn_class_adaptive_v1
    bash submit.sh start
)
```

This submits one batch job. It checks package files, runs 81 tests in the pinned
environment, runs a real-image GPU smoke, verifies its gate, then trains for
20 epochs. Queued means it has not started. A disconnect does not terminate it.

## 3. CCAST login node: read status and latest output

```bash
(
    set -euo pipefail
    class_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    class_job=$(cat "$class_project/fasterrcnn_class_adaptive_v1/submission_start.id")
    qstat "$class_job" || true
    class_log="$class_project/runs/baseline_job_logs/$class_job.log"

    if [ -f "$class_log" ]; then
        tail -n 80 "$class_log"
    else
        printf 'No live log yet; the job may still be queued.\n'
    fi
)
```

This is read-only and can be repeated. Missing `qstat` history alone does not
establish success; inspect the log and saved `run.json`. Do not resubmit `start`
after a disconnect. If the run stops on its time budget, inspect the checkpoint
and completed epochs before explicitly resuming after the previous job has ended.
Download instructions will use the actual records archive name and checksum
printed at completion. No result archive or job ID is invented in advance.
