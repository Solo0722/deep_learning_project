& {
    $ErrorActionPreference = "Stop"
    $pair8Bundle = "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\rdd_v2_pair8_v1.tar.gz"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $pair8Bundle).Hash -ne "fbebbf012aa24802b1c96362d54df6463a51c787900e7106c13365486285b533") {
        throw "Package checksum mismatch."
    }
    scp $pair8Bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) { throw "Upload failed." }
}
