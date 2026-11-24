# RDD v2 preflight: commands

This release performs CPU annotation/image checks. It does not train a detector.
The environment is activated inside the PBS job; no interactive GPU allocation
or login-shell environment activation is needed.

Run this block in **Windows PowerShell**:

```powershell
& {
    $ErrorActionPreference = "Stop"
    $bundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\rdd_v2_preflight_v1.tar.gz"
    $expected = "c4fe916b27c0215ff8c0681b6d0f4723219e21817786649c3988a54699849f4d"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash -ne $expected) {
        throw "Package checksum mismatch."
    }
    scp $bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
}
```

Then, on a **CCAST login node**, run this block:

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\n' \
      'c4fe916b27c0215ff8c0681b6d0f4723219e21817786649c3988a54699849f4d' \
      'rdd_v2_preflight_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_preflight_v1 ]; then
        printf 'Package directory already exists; inspect it before continuing.\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_preflight_v1.tar.gz
    cd rdd_v2_preflight_v1
    bash submit.sh
)
```

To inspect the submitted job later:

```bash
(
    set -euo pipefail
    preflight_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    preflight_job=$(cat "$preflight_project/rdd_v2_preflight_v1/submission_start.id")
    qstat "$preflight_job" || true
    preflight_log="$preflight_project/runs/preflight_job_logs/$preflight_job.log"
    if [ -f "$preflight_log" ]; then
        tail -n 60 "$preflight_log"
    else
        printf 'No live log yet; the job may still be queued.\n'
    fi
)
```

The completion log reports the checksum for
`runs/rdd_v2_geometry_check_v1.tar.gz`. Download that result and verify its
reported checksum before reviewing the overlays. Do not resubmit after an SSH
disconnect; use the saved job ID to inspect the existing batch job.
