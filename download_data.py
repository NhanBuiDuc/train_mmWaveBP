"""Download the public data sets that need no account (standard library only).

    python download_data.py --list
    python download_data.py erlangen --out data
    python download_data.py airbp mimic_bp --out data

Files are resumed when partly present and unzipped into <out>/<name>/. Data sets behind a free
login (Blumio wrist radar, VitalSense, bed BCG) cannot be fetched by a script: `--list` prints
their pages; download them in a browser and unzip them into <out>/<name>/ yourself.
Read each licence in README.md before using or redistributing a data set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

GB = 1e9
DIRECT = {
    # name: (description, [(file name, url, size in bytes or None)])
    "erlangen": ("CHEST  Erlangen phase 1: 24 GHz radar + continuous BP, 30 subjects, CC BY 4.0", [
        ("datasets_subject_01_to_10_scidata.zip", "https://ndownloader.figshare.com/files/22515785", 2090795424),
        ("datasets_subject_11_to_20_scidata.zip", "https://ndownloader.figshare.com/files/22516136", 1946328412),
        ("datasets_subject_21_to_30_scidata.zip", "https://ndownloader.figshare.com/files/22521941", 1972962229),
        ("additional_data.xlsx", "https://ndownloader.figshare.com/files/22515782", 21363)]),
    "zenodo110": ("CHEST  110 participants, 60 GHz range FFT at 10 Hz, one cuff reading each, CC BY 4.0", [
        ("db_records.zip", "https://zenodo.org/api/records/18599983/files/db_records.zip/content", 245.3e6),
        ("ParticipantsInfo.xlsx", "https://zenodo.org/api/records/18599983/files/ParticipantsInfo.xlsx/content", None)]),
    "trccbp": ("CHEST  TRCCBP: 7.3 GHz UWB at 20 Hz, cuff value per recording, NO LICENCE FILE", [
        ("TRCCBP-master.zip", "https://github.com/bupt-uwb/TRCCBP/archive/refs/heads/master.zip", 1.69e9)]),
    "erlangen_heartsounds": ("CHEST  same radar as Erlangen, 11 subjects, no BP (pre-training), CC BY 4.0", [
        ("datasets_scidata_vsmdb.zip", "https://ndownloader.figshare.com/files/17357702", 583572264)]),
    "airbp": ("WRIST  airBP public sample: 15 clips at 210 Hz, cuff values, MIT", [
        ("dataset_blood_pressure.zip",
         "https://raw.githubusercontent.com/YumengLiang/dataset-of-mmwave/main/dataset_blood_pressure.zip", 737e3)]),
    "polypulse": ("CHEST + WRIST  PolyPulse example: one user, several body sites, no BP labels, CC BY 4.0", [
        ("data.zip", "https://zenodo.org/api/records/19403782/files/data.zip/content", 5751498457)]),
    "mimic_bp": ("PRE-TRAINING  MIMIC-BP: PPG / ECG + invasive BP, 1524 subjects, ODbL 1.0", [
        ("abp.zip", "https://dataverse.harvard.edu/api/access/datafile/7574731", 245e6),
        ("ppg.zip", "https://dataverse.harvard.edu/api/access/datafile/7574732", 276e6),
        ("ecg.zip", "https://dataverse.harvard.edu/api/access/datafile/7574838", 156e6),
        ("resp.zip", "https://dataverse.harvard.edu/api/access/datafile/7574839", 243e6),
        ("labels", "https://dataverse.harvard.edu/api/access/datafile/7574727", None)]),
    "aurora_bp": ("PRE-TRAINING  Aurora-BP: wrist tonometry + cuff BP, Microsoft data use agreement", [
        ("measurements_oscillometric.zip",
         "https://zenodo.org/api/records/19099166/files/measurements_oscillometric.zip/content", 5.0e9),
        ("measurements_auscultatory.zip",
         "https://zenodo.org/api/records/19099166/files/measurements_auscultatory.zip/content", 1.6e9)]),
    "pwdb": ("PRE-TRAINING  Pulse Wave Database: 4374 simulated subjects, ODC PDDL", [
        ("PWs_csv.zip", "https://zenodo.org/api/records/3275625/files/PWs_csv.zip/content", 243e6)]),
    "ppg_bp": ("PRE-TRAINING  PPG-BP: 219 subjects, fingertip PPG + cuff, CC0", [
        ("ppg_bp.zip", "https://ndownloader.figshare.com/files/9441097", 1.5e6)]),
}
LOGIN = {
    "blumio": ("WRIST  Blumio: 60 GHz wearable radar + continuous BP, 115 subjects (free IEEE account)",
               "https://ieee-dataport.org/open-access/dataset-synchronized-signals-wearable-cardiovascular-monitoring-sensors"),
    "vitalsense": ("CHEST  VitalSense 120 GHz: 24 subjects, 2 cuff readings each (free IEEE account)",
                   "https://ieee-dataport.org/open-access/new-dataset-millimeter-wave-radar-vital-sensing-reference-signals"),
    "bed_bcg": ("PRE-TRAINING  Bed ballistocardiography + continuous BP, 40 subjects (free IEEE account)",
                "https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset"),
}


def fetch(url: str, target: Path) -> None:
    """Download with resume (HTTP Range) and a progress line."""
    have = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "train_mmWaveBP", **({"Range": f"bytes={have}-"} if have else {})})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:                                           # already complete
            return
        raise
    resumed = response.status == 206
    total = int(response.headers.get("Content-Length", 0)) + (have if resumed else 0)
    done = have if resumed else 0
    with response, open(target, "ab" if resumed else "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {target.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of: {', '.join(DIRECT)}")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-zip", action="store_true", help="do not delete a zip after extracting it")
    args = ap.parse_args()
    if args.list or not args.names:
        print("direct download:")
        for name, (text, files) in DIRECT.items():
            size = sum(s or 0 for _, _, s in files)
            print(f"  {name:<22} {size / GB:6.2f} GB  {text}")
        print("\nbrowser download (login):")
        for name, (text, page) in LOGIN.items():
            print(f"  {name:<22} {text}\n  {'':<22} {page}")
        return 0
    for name in args.names:
        if name in LOGIN:
            print(f"{name}: needs a login, open {LOGIN[name][1]}")
            continue
        if name not in DIRECT:
            print(f"{name}: unknown (see --list)")
            return 1
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        files = DIRECT[name][1]
        need = sum(s or 0 for _, _, s in files)
        if shutil.disk_usage(folder).free < 2.2 * need:
            print(f"{name}: needs about {2.2 * need / GB:.1f} GB free in {folder} (zip + extracted)")
            return 1
        for filename, url, _ in files:
            target = folder / filename
            fetch(url, target)
            if zipfile.is_zipfile(target):
                print(f"  extracting {filename} ...")
                with zipfile.ZipFile(target) as z:
                    z.extractall(folder)
                if not args.keep_zip:
                    target.unlink()
        print(f"{name}: ready in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
