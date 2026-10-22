#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/REFEREE"
source .venv/bin/activate

git var GIT_AUTHOR_IDENT
git diff --cached --quiet || {
    echo "Ada perubahan di staging. Selesaikan dahulu."
    exit 1
}

export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/referee_github -o IdentitiesOnly=yes -o BatchMode=yes"
REMOTE="git@github.com:amri-muhaimin-au/REFEREE.git"
git push --dry-run "$REMOTE" HEAD:main

python - <<'PY'
from pathlib import Path

base = Path.home() / "datasets"
for relative, expected in [
    ("domainnet/clipart", 345),
    ("domainnet/sketch", 345),
    ("visda/train", 12),
    ("visda/validation", 12),
]:
    folder = base / relative
    assert folder.is_dir(), f"Folder tidak ditemukan: {folder}"
    count = sum(p.is_dir() for p in folder.iterdir())
    assert count == expected, f"{folder}: {count}, seharusnya {expected}"
    print(f"OK: {relative}, {count} kelas")
PY

for DATASET in DomainNet VisDA; do
    RUN="experiments/${DATASET}_D256_workers0_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN"
    export REFEREE_DATASET="$DATASET"
    export REFEREE_RUN_DIR="$PWD/$RUN"

    git rev-parse HEAD > "$RUN/source_commit.txt"
    python -m pip freeze > "$RUN/dependencies.txt"
    cp "$HOME/run_referee_large.sh" "$RUN/launcher.sh"

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

# Tetap berupa class agar kompatibel dengan timm.
torch.multiprocessing.set_sharing_strategy("file_system")
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

# Alihkan path dataset hanya saat runtime.
original_folder = datasets.ImageFolder

def image_folder(root, *args, **kwargs):
    root = str(root)
    for prefix, folder in [
        ("/newEra/furqon/DomainNet/", "domainnet"),
        ("/newEra/furqon/data/visda/", "visda"),
    ]:
        if root.startswith(prefix):
            root = str(base / folder / root[len(prefix):])
            break
    return original_folder(root, *args, **kwargs)

datasets.ImageFolder = image_folder

verified_context = ssl.create_default_context
import data
ssl._create_default_https_context = verified_context

# Periksa splitting dari kode repository yang telah diperbaiki.
loader = (
    data.get_subset_domainnet
    if name == "DomainNet"
    else data.get_subset_visda
)
original_subset = loader.subset

def checked_subset(self, *args, **kwargs):
    train, test = original_subset(self, *args, **kwargs)
    assert set(train.index).isdisjoint(test.index), "Overlap train/test!"
    print(f"[Split check] {self.root_dir}: overlap = 0")
    return train, test

loader.subset = checked_subset

# Verifikasi kompatibilitas sebelum memulai eksperimen.
import timm
probe = SingleProcessDataLoader(
    torch.utils.data.TensorDataset(torch.arange(4)),
    batch_size=2,
    num_workers=4,
)
assert probe.num_workers == 0
assert sum(len(batch[0]) for batch in probe) == 4
print("[Loader check] timm berhasil diimpor; workers=0")

assert torch.cuda.is_available(), "GPU belum terdeteksi"
print(f"[GPU] {torch.cuda.get_device_name(0)}")
print(f"[Experiment] {name}, D=256, batch=2, seed=0, workers=0")

os.chdir(os.environ["REFEREE_RUN_DIR"])
sys.argv = [
    "evaluate.py", "--source", name,
    "--scenario", "class", "--seed", "0",
    "--runs", "1", "--batch_size", "2",
    "--D", "256", "--num_ensembles", "1",
]
runpy.run_path(str(repo / "evaluate.py"), run_name="__main__")
PY

    EXPECTED=2
    if [ "$DATASET" = VisDA ]; then EXPECTED=1; fi

    COUNT=$(grep -c '^Final Average Target Accuracy:' "$RUN/run.log")
    if [ "$COUNT" -ne "$EXPECTED" ]; then
        echo "Ringkasan tidak lengkap: $COUNT/$EXPECTED. Berhenti."
        exit 1
    fi

    git add -- "$RUN"
    if git commit --only \
        -m "Record ${DATASET} D256 seed 0 workers 0 experiment" \
        -- "$RUN"; then
        if git push "$REMOTE" HEAD:main; then
            echo "SELESAI DAN TERPUSH: $DATASET"
        else
            echo "PUSH GAGAL: hasil tersimpan lokal; lanjut dataset berikutnya."
        fi
    else
        echo "COMMIT GAGAL: hasil tersimpan di $RUN; lanjut dataset berikutnya."
    fi
done

echo "SEMUA EKSPERIMEN SELESAI. Periksa status push di atas."
