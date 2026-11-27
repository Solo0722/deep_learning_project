# CCAST four-arm pilot commands

Run block 1 in Windows PowerShell. Run block 2 in your existing CCAST login-node SSH session. No interactive GPU request or manual environment activation is needed. The PBS job runs the tests, GPU gate and bounded pilot sequentially.

## 1. Upload from Windows PowerShell

```powershell
& {
    $ErrorActionPreference = "Stop"
    $bundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\rdd_v2_pilot_v1.tar.gz"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash -ne "92b7ff1170919dc6beed2aadff44aa147f7230580381ee67d6d56a21d4d57b7a") {
        throw "Package checksum mismatch."
    }
    scp $bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
}
```

## 2. Verify, extract and submit on the CCAST login node

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\n' '92b7ff1170919dc6beed2aadff44aa147f7230580381ee67d6d56a21d4d57b7a' 'rdd_v2_pilot_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_pilot_v1 ]; then
        printf 'Package directory already exists; stop and inspect it.\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_pilot_v1.tar.gz
    cd rdd_v2_pilot_v1
    bash submit.sh start
)
```

## 3. Status and recent log

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pilot_v1/status.sh
```

The single batch allocation requests one A10, four CPUs and 32 GB RAM, with a 12-hour PBS ceiling and an 11-hour administrative cap. Each arm runs epochs 2–4 from the same epoch-1 state. The controller updates before epochs 2 and 4. A disconnect does not cancel the batch job. Do not submit a duplicate; reconnect and use block 3.

If the run stops at its time cap, inspect the log and status before explicitly using `bash submit.sh resume` from the package directory. This resumes the same bounded pilot, not a longer experiment. Failed tests or the smoke gate must be inspected, not bypassed.

Download the exact archive printed by the completed job and verify its printed SHA-256. The source archive and result archive have different filenames. Full checkpoint tensors remain on CCAST. No new pilot has been run locally or remotely by preparing this package.
