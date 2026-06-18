from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from typing import List, Optional

import requests

try:
    from tqdm import tqdm
except Exception:  # tqdm не обязателен
    def tqdm(x, **k):
        return x

# Публичный NBIA REST API (для открытых коллекций авторизация не нужна).
DEFAULT_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_series(collection: str, base_url: str = DEFAULT_BASE_URL,
               modality: Optional[str] = None, timeout: int = 60) -> List[dict]:
    """Список серий коллекции. Возвращает записи с SeriesInstanceUID, PatientID, Modality."""
    params = {"Collection": collection, "format": "json"}
    if modality:
        params["Modality"] = modality
    r = requests.get(f"{base_url}/getSeries", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_series(series_uid: str, dest_dir: str, base_url: str = DEFAULT_BASE_URL,
                    retries: int = 3, timeout: int = 600) -> bool:
    """Скачать одну серию (zip с DICOM) и распаковать в dest_dir. True, если успешно."""
    os.makedirs(dest_dir, exist_ok=True)
    # если уже скачано (есть .dcm файлы) — пропускаем
    if any(f.lower().endswith(".dcm") for f in os.listdir(dest_dir)):
        return True
    url = f"{base_url}/getImage"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, params={"SeriesInstanceUID": series_uid},
                              stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(dest_dir)
            return True
        except Exception as e:
            print(f"  [попытка {attempt}/{retries}] ошибка для {series_uid}: {e}",
                  file=sys.stderr)
            time.sleep(2 * attempt)
    return False


def download_collection(collection: str, out: str, base_url: str = DEFAULT_BASE_URL,
                        modality: Optional[str] = None, max_series: Optional[int] = None) -> None:
    print(f"=== {collection}: запрашиваю список серий ===")
    series = get_series(collection, base_url=base_url, modality=modality)
    if max_series:
        series = series[:max_series]
    print(f"Серий к загрузке: {len(series)}")

    ok, fail = 0, 0
    for s in tqdm(series, desc=collection):
        uid = s["SeriesInstanceUID"]
        patient = s.get("PatientID", "unknown")
        dest = os.path.join(out, collection, patient, uid)
        if download_series(uid, dest, base_url=base_url):
            ok += 1
        else:
            fail += 1
    print(f"Готово: успешно {ok}, ошибок {fail}. Данные в {os.path.join(out, collection)}")


def main():
    ap = argparse.ArgumentParser(description="Скачивание коллекций TCIA (NBIA REST API).")
    ap.add_argument("--collection", required=True,
                    help="например HCC-TACE-Seg или Colorectal-Liver-Metastases")
    ap.add_argument("--out", default="/data/tcia", help="корневая папка для данных")
    ap.add_argument("--modality", default=None, help="фильтр модальности: CT | SEG | ...")
    ap.add_argument("--max-series", type=int, default=None, help="ограничить число серий (тест)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args()

    download_collection(args.collection, args.out, base_url=args.base_url,
                        modality=args.modality, max_series=args.max_series)


if __name__ == "__main__":
    main()
