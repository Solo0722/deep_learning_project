# Epoch-8 paired experiment

Local validation passed; the CCAST GPU gate and accuracy run are pending. No remote submission has been performed.

1. Run in Windows PowerShell:

```powershell
& {
    $ErrorActionPreference = "Stop"
    $pair8Bundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\rdd_v2_pair8_v1.tar.gz"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $pair8Bundle).Hash -ne "fbebbf012aa24802b1c96362d54df6463a51c787900e7106c13365486285b533") {
        throw "Package checksum mismatch."
    }
    scp $pair8Bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
}
```

2. Run on a CCAST login node after the upload completes. The batch job activates its environment and GPU allocation.

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\n' 'fbebbf012aa24802b1c96362d54df6463a51c787900e7106c13365486285b533' 'rdd_v2_pair8_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_pair8_v1 ]; then
        printf 'Package directory already exists; inspect it before continuing.\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_pair8_v1.tar.gz
    cd rdd_v2_pair8_v1
    bash submit.sh start
)
```

3. Check the job and recent log:

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pair8_v1/status.sh
```

The job runs its exactness gate, then class-adaptive and geometry-only training through epoch 8. An 11-hour deadline checkpoints and stops; inspect that output before explicit resume. No automatic resubmission. SSH may disconnect without cancelling the batch job.
