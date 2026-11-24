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
