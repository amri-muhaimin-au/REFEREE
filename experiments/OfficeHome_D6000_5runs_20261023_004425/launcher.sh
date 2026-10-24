#!/usr/bin/env bash
set -euo pipefail

# Pastikan eksperimen terlindungi tmux.
if [ -z "${TMUX:-}" ]; then
    echo "Berhenti: masuk ke tmux dahulu."
    exit 1
fi

cd "$HOME/REFEREE"
source .venv/bin/activate

git var GIT_AUTHOR_IDENT
git diff --quiet || {
    echo "Ada perubahan kode lokal yang belum di-commit."
    exit 1
}
git diff --cached --quiet || {
    echo "Ada perubahan di staging. Selesaikan dahulu."
    exit 1
}

export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/referee_github -o IdentitiesOnly=yes -o BatchMode=yes"
REMOTE="git@github.com:amri-muhaimin-au/REFEREE.git"
git push --dry-run "$REMOTE" HEAD:main

python - <<'PY'
from pathlib import Path
import torch

base = Path.home() / "datasets"
folders = [
    ("office31/amazon/images", 31),
    ("office31/dslr/images", 31),
    ("office31/webcam/images", 31),
    ("officehome/Art", 65),
    ("officehome/Clipart", 65),
    ("officehome/Product", 65),
    ("officehome/Real_World", 65),
    ("domainnet/clipart", 345),
    ("domainnet/sketch", 345),
    ("visda/train", 12),
    ("visda/validation", 12),
]
for relative, expected in folders:
    folder = base / relative
    assert folder.is_dir(), f"Folder tidak ditemukan: {folder}"
    count = sum(p.is_dir() for p in folder.iterdir())
    assert count == expected, f"{folder}: {count}, seharusnya {expected}"
    print(f"OK: {relative}, {count} kelas")

assert torch.cuda.is_available(), "CUDA tidak tersedia"
print("GPU:", torch.cuda.get_device_name(0))
PY

for DATASET in Office31 OfficeHome DomainNet VisDA; do
    RUN="experiments/${DATASET}_D6000_5runs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN"

    export REFEREE_DATASET="$DATASET"
    export REFEREE_RUN_DIR="$PWD/$RUN"

    git rev-parse HEAD > "$RUN/source_commit.txt"
    python -m pip freeze > "$RUN/dependencies.txt"
    cp "$HOME/run_referee_full.sh" "$RUN/launcher.sh"
    nvidia-smi > "$RUN/gpu_before.txt"

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
    python -u - <<'PY' 2>&1 | tee "$RUN/run.log"
import os
import sys
import ssl
import runpy
from pathlib import Path

import torch
import torch.utils.data
from torchvision import datasets

repo = Path.home() / "REFEREE"
base = Path.home() / "datasets"
name = os.environ["REFEREE_DATASET"]
sys.path.insert(0, str(repo))

# DataLoader tetap class, tetapi berjalan tanpa worker tambahan.
OriginalDataLoader = torch.utils.data.DataLoader

class SingleProcessDataLoader(OriginalDataLoader):
    def __init__(self, *args, **kwargs):
        positional = list(args)
        if len(positional) > 5:
            positional[5] = 0
            kwargs.pop("num_workers", None)
        else:
            kwargs["num_workers"] = 0

        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = None
        kwargs["multiprocessing_context"] = None
        super().__init__(*positional, **kwargs)

torch.utils.data.DataLoader = SingleProcessDataLoader

# Pengalihan lokasi dataset tanpa mengedit repository.
original_folder = datasets.ImageFolder

def image_folder(root, *args, **kwargs):
    root = str(root)
    mappings = [
        ("/newEra/furqon/data/office/", "office31"),
        ("/newEra/furqon/data/OfficeHome/", "officehome"),
        ("/newEra/furqon/DomainNet/", "domainnet"),
        ("/newEra/furqon/data/visda/", "visda"),
    ]
    for prefix, folder in mappings:
        if root.startswith(prefix):
            relative = root[len(prefix):]
            if folder == "officehome" and relative == "Real World":
                relative = "Real_World"
            root = str(base / folder / relative)
            break
    return original_folder(root, *args, **kwargs)

datasets.ImageFolder = image_folder

verified_context = ssl.create_default_context
import data
ssl._create_default_https_context = verified_context

loader = {
    "Office31": data.get_subset_office31,
    "OfficeHome": data.get_subset_officeHome,
    "DomainNet": data.get_subset_domainnet,
    "VisDA": data.get_subset_visda,
}[name]

original_subset = loader.subset

def checked_subset(self, *args, **kwargs):
    train, test = original_subset(self, *args, **kwargs)
    assert set(train.index).isdisjoint(test.index), "Overlap train/test!"
    print(f"[Split check] {self.root_dir}: overlap = 0")
    return train, test

loader.subset = checked_subset

import timm
probe = SingleProcessDataLoader(
    torch.utils.data.TensorDataset(torch.arange(4)),
    batch_size=2,
    num_workers=4,
)
assert probe.num_workers == 0
assert sum(len(batch[0]) for batch in probe) == 4

print("[Loader check] timm OK; workers=0")
print(f"[Experiment] {name}, D=6000, batch=2, seeds=0,1,2,3,4")

os.chdir(os.environ["REFEREE_RUN_DIR"])
sys.argv = [
    "evaluate.py", "--source", name,
    "--scenario", "class",
    "--seed", "0", "--runs", "5",
    "--batch_size", "2",
    "--D", "6000", "--num_ensembles", "1",
]
runpy.run_path(str(repo / "evaluate.py"), run_name="__main__")
PY

    case "$DATASET" in
        Office31) EXPECTED=30 ;;
        OfficeHome) EXPECTED=60 ;;
        DomainNet) EXPECTED=10 ;;
        VisDA) EXPECTED=5 ;;
    esac

    COUNT=$(grep -c '^Final Average Target Accuracy:' "$RUN/run.log")
    if [ "$COUNT" -ne "$EXPECTED" ]; then
        echo "Hasil tidak lengkap: $COUNT/$EXPECTED. Berhenti."
        exit 1
    fi

    git add -- "$RUN"
    if git commit --only \
        -m "Record ${DATASET} D6000 five-seed experiment" \
        -- "$RUN"; then
        if git push "$REMOTE" HEAD:main; then
            echo "SELESAI DAN TERPUSH: $DATASET"
        else
            echo "Push gagal; hasil tersimpan lokal. Lanjut dataset berikutnya."
        fi
    else
        echo "Commit gagal; hasil tersimpan di $RUN. Lanjut dataset berikutnya."
    fi
done

echo "SEMUA EKSPERIMEN SELESAI. Periksa status push di atas."
