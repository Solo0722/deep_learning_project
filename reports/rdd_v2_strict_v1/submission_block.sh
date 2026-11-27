(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\n' 'f1698acba680d90a08368a570f5f2debdb89e262cc5d902be3581ec6f9d8b784' 'rdd_v2_strict_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_strict_v1 ]; then
        printf 'Package directory already exists; inspect it before continuing.\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_strict_v1.tar.gz
    cd rdd_v2_strict_v1
    bash submit.sh
)
