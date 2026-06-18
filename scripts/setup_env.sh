#!/usr/bin/env bash
# Создание окружения на удалённой GPU-машине (зеркало рабочего segvol_env).
# Python 3.10 + torch 1.13.1 (CUDA 11.7) + зависимости пайплайна.
#
# Использование:
#   bash scripts/setup_env.sh            # создаст venv ./segvol_env и всё поставит
#   source segvol_env/bin/activate       # активировать перед работой
#
# Если на машине новый GPU и драйвер не поддерживает CUDA 11.7 — см. блок FALLBACK ниже.

set -euo pipefail

ENV_DIR="${1:-segvol_env}"
PYBIN="${PYTHON:-python3.10}"

echo "=== 1/4 Проверка Python ==="
if ! command -v "$PYBIN" >/dev/null 2>&1; then
    echo "Нужен python3.10. Установите: sudo apt install python3.10 python3.10-venv" >&2
    exit 1
fi
"$PYBIN" --version

echo "=== 2/4 Создание venv: $ENV_DIR ==="
"$PYBIN" -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "=== 3/4 Установка torch 1.13.1 + cu117 ==="
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

echo "=== 4/5 Установка зависимостей пайплайна ==="
pip install -r requirements-train.txt

echo "=== 5/5 Установка pyradiomics (опционально, отдельным шагом) ==="
# Сборка pyradiomics через pip падает из-за изоляции build-env (нет numpy).
# Ставим build-зависимости в сам venv и отключаем изоляцию.
pip install "cython<3" numpy==1.26.4
if pip install pyradiomics==3.0.1 --no-build-isolation; then
    echo "pyradiomics установлен."
else
    echo "!! pyradiomics не собрался. Пайплайн работает и без него (graceful skip)."
    echo "   Фолбэк 1 (git, с фиксами под numpy>=1.24):"
    echo "     pip install \"cython<3\" --no-build-isolation git+https://github.com/AIM-Harvard/pyradiomics.git"
    echo "   Фолбэк 2 (conda, надёжнее):  conda install -c conda-forge pyradiomics"
fi

echo "=== Проверка ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
import monai, transformers, nibabel, SimpleITK, sklearn, pandas
print("monai", monai.__version__, "| transformers", transformers.__version__)
try:
    import radiomics; print("pyradiomics", radiomics.__version__)
except Exception as e:
    print("pyradiomics НЕ установлен:", e)
PY

cat <<'NOTE'

=== Готово ===
Активировать окружение:    source segvol_env/bin/activate
Проверить пайплайн (CPU):  python -m liver_sppvr.train.multitask --config configs/default.yaml --dry-run --device cpu

--- FALLBACK: если torch+cu117 не подходит под GPU/драйвер ---
Удалите строку torch выше и поставьте новее под вашу CUDA, например:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
(monai 0.9.0 со свежим torch иногда конфликтует — тогда поднимите monai до 1.3.x.)

--- FALLBACK: если pyradiomics не собирается через pip ---
Поставьте через conda-forge (надёжнее на Python 3.10):
    conda install -c conda-forge pyradiomics
NOTE
