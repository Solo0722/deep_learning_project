# RDD v2 controller diagnostic: command blocks

Prepared package: `rdd_v2_controller_check_v1.tar.gz` (47,820 bytes).
SHA-256: `3a101b03d367c3222803c4a7ca85b2473d2cfaca115ea70d940ec07463e06690`.

Local validation: 16 tests passed; two PyTorch-dependent observer tests skipped
locally and required on CCAST. All Python files compile, all three Bash scripts
pass syntax checks, all 16 internal file checksums match the archive. Local
execution used Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0. The remote job
requires the pinned CCAST versions before real inference. GPU/raw-image checks
have not run yet. The previous experiments and preflight files are unchanged.

The next evidence is support and two-panel policy sensitivity at a common
frozen first-epoch detector. It is not AP or a training run. Each panel has 64
anchor groups per class. The selected groups contain D40 objects numbering 91
and 95 in panels A and B, respectively; these are annotation counts, not detected
support. There are 512 images and 512 distinct source clusters across the two
panels. The additional thin Norway case is outside both panels.

The checkpoint metadata was retrieved from the verified identity records
archive, not selected using validation AP. Its actual presence and bytes on
CCAST will be checked before submission. If it is missing, stop and inspect
the retained checkpoints; do not substitute the best checkpoint.

## 1. Upload from Windows PowerShell

```powershell
& {
    $ErrorActionPreference = "Stop"
    $bundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\rdd_v2_controller_check_v1.tar.gz"
    $expected = "3a101b03d367c3222803c4a7ca85b2473d2cfaca115ea70d940ec07463e06690"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash -ne $expected) {
        throw "Package checksum mismatch."
    }
    scp $bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
}
```

## 2. Submit from a CCAST login terminal

No interactive GPU allocation or manual environment activation is needed.
If disconnected, connect using `ssh solomon.owusu.1@prime.ccast.ndsu.edu` first.

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\n' \
      '3a101b03d367c3222803c4a7ca85b2473d2cfaca115ea70d940ec07463e06690' \
      'rdd_v2_controller_check_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_controller_check_v1 ]; then
        printf 'Package directory already exists. Stop and inspect; do not resubmit.\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_controller_check_v1.tar.gz
    cd rdd_v2_controller_check_v1
    bash submit.sh
)
```

This submits one A10 GPU, four CPUs, 32 GB and one hour. The job activates the
existing environment, runs all 18 tests, then collects the bounded diagnostic.
An SSH disconnect does not cancel the batch job. The script refuses duplicate
submissions and existing results. No training follows automatically.

## 3. Check status and the saved log

```bash
(
    set -euo pipefail
    controller_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    controller_job=$(cat "$controller_project/rdd_v2_controller_check_v1/submission_start.id")
    qstat "$controller_job" || true
    controller_log="$controller_project/runs/controller_job_logs/$controller_job.log"
    if [ -f "$controller_log" ]; then
        tail -n 80 "$controller_log"
    else
        printf 'No live log yet; the job may still be queued.\n'
    fi
)
```

On completion, the job prints the records archive path and SHA-256. Download
and verify that archive before auditing the support, stability and previews.
The path will be `runs/rdd_v2_controller_check_v1.tar.gz`. A complete diagnostic
is a review gate, not proof that the method improves accuracy. The next training
stage, if warranted, is the bounded matched four-arm pilot.
