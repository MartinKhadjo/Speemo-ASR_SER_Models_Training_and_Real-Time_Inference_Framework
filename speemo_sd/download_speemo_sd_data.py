import os
from pathlib import Path
import zipfile

import gdown  # pip install gdown

# HIER deine Google-Drive File-IDs eintragen:
DRIVE_FILES = {
    "data": {
        "id": "18eidKqodyWlZfyifGiBSDPKOTIsXfpbq",      # z.B. 1AbCdEf...
        "zip_name": "data.zip",
        "extract_to": "data",
    },
    "models": {
        "id": "1e03i9DGXTj0OxH5rikZxtwTz_vc7YAZV",    # z.B. 1XyZ123...
        "zip_name": "models.zip",
        "extract_to": "models",
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_and_extract(name: str, info: dict, force_redownload: bool = False):
    file_id = info["id"]
    zip_name = info["zip_name"]
    extract_to = Path(info["extract_to"])

    ensure_dir(extract_to)

    zip_path = Path(zip_name)

    if zip_path.exists() and not force_redownload:
        print(f"[{name}] ZIP existiert schon: {zip_path} – überspringe Download")
    else:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[{name}] Lade von Google Drive...")
        gdown.download(url, str(zip_path), quiet=False)  # zeigt Progress-Bar
        print(f"[{name}] Download fertig: {zip_path}")

    print(f"[{name}] Entpacke nach: {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[{name}] Entpacken fertig.\n")


def main():
    # Projekt-Root = Verzeichnis dieser Datei
    os.chdir(Path(__file__).resolve().parent)

    # Reihenfolge ist egal, nur der Name für die Logs
    for name, info in DRIVE_FILES.items():
        download_and_extract(name, info, force_redownload=False)


if __name__ == "__main__":
    main()
