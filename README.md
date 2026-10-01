# train_mmWaveBP

Training code for contactless (mmWave / RF radar) blood-pressure models, and the list of public
data sets to train them on. Training only: no radar capture, no live demo.

| Method | Paper | Body site | Model | File |
|---|---|---|---|---|
| WaveBP | Hu et al., IMWUT 2024 | chest | U-Net + Transformer (mmFormer), 5 AHA experts, BP waveform output | `mmwave_bp/methods/wavebp.py` |
| RF-BP | Wang et al., IMWUT 2024 | chest | multi-scale conv + 8 attention residual blocks | `mmwave_bp/methods/rfbp.py` |
| airBP | Liang et al., ACM TIoT 2023 | wrist | masked auto-encoder pre-training + conv self-attention | `mmwave_bp/methods/airbp.py` |
| mmBP | Shi et al., SenSys 2022 | wrist | TR-FLAF + 7 beat features + random forest | `mmwave_bp/methods/mmbp.py` |
| hBP-Fi | Cao et al., INFOCOM 2024 | arm, several sites | ADMM-unrolled DRCN + tube-load model + LSTM | `mmwave_bp/methods/hbpfi.py` |

None of the five papers released code or (except 15 airBP clips) data. The models here are
written from the papers. Each file's docstring lists what comes from the paper and what had to
be chosen because the paper does not say. Two of them are only partly reproducible:

* **mmBP**: the delay-Doppler transform (DDFT) is not specified well enough to rebuild and is
  replaced by a band-pass filter. TR-FLAF is an interpretation.
* **hBP-Fi**: the paper gives the network as a block diagram only (no sizes, no ADMM matrices,
  no constants). The file is an interpretation with the paper's structure, not a reproduction.

## Install and check

```bash
pip install -r requirements.txt
```

```bash
python selftest.py
```

`selftest.py` trains every method for one epoch on synthetic pulses (numbers are meaningless,
it only shows the code runs).

## Train

```bash
python download_data.py erlangen --out data
```

```bash
python train.py --method wavebp --dataset erlangen --data data/erlangen --out runs/wavebp --epochs 50 --flip --final
```

```bash
python train.py --method rfbp --dataset erlangen --data data/erlangen --out runs/rfbp --epochs 40 --final
```

```bash
python train.py --method airbp --dataset blumio --data data/blumio --out runs/airbp --pretrain-epochs 30 --epochs 100 --final
```

```bash
python train.py --method mmbp --dataset blumio --data data/blumio --out runs/mmbp --final
```

```bash
python train.py --method hbpfi --dataset npz --data data/my_arm_sites --out runs/hbpfi --final
```

* Folds are made of whole subjects; a model is always scored on people it never saw.
* A stopped run continues with the same command (per-epoch checkpoints, finished folds kept).
* Part of the training subjects (`--val-fraction`, default 15 %) picks the best epoch, so a
  long run does not end on over-fitted weights.
* Paper-size WaveBP: `--set d_model=256 layers=12`. Any model argument can be set this way.
* airBP window preparation runs VMD (about 1 s per 30 s window); the windows are cached in the
  run folder.

### Which method can use which data set

| | `erlangen` (chest, continuous BP) | `blumio` (wrist, continuous BP) | `airbp` (wrist, 15 clips, cuff) | `npz` (your data) |
|---|---|---|---|---|
| wavebp | yes | yes | no (needs a BP waveform) | with `abp` |
| rfbp | yes | yes | no (its breathing-stationarity gate rejects the 30 s wrist clips) | yes |
| airbp | yes | yes | yes | yes |
| mmbp | yes | yes | yes | yes |
| hbpfi | no | no | no | with `pulse` of shape [sites, n], sites >= 2 |

Tested here on the real files: `erlangen` with every single-site method (WaveBP and RF-BP
trained for a few epochs) and `airbp` (airBP trained leave-one-subject-out on the 15 clips: no
better than the baselines, as expected from 5 subjects). **Not tested: the `blumio` loader**
(written from the published file description; check the columns on the first file). hBP-Fi has
no public data at all.

Your own recordings: one `.npz` per recording with `pulse`, `fs`, `subject` and either
`abp` + `fs_abp` or `sbp` + `dbp` (details in `mmwave_bp/data.py`, function `npz`).

### Reading the result

`train.py` prints, and writes to `runs/<name>/report.json`:

| Row | Meaning |
|---|---|
| model | the network alone |
| model + offset | after one cuff reading per person (mean error on the calibration windows removed) |
| B1 population mean | everybody gets the training subjects' mean BP |
| B0 carry calibration | the calibration reading repeated |

A model is useful without calibration only if it beats **B1**, and with calibration only if it
beats **B0**. `r` is the within-subject correlation: whether the estimate follows a person's BP
changes. On Erlangen (30 healthy young subjects) our WaveBP runs so far did **not** beat these
baselines on unseen people; a low training loss says nothing about that. More and more varied
subjects are the limiting factor, which is why the data list below matters.

## Public data sets

Checked on 2026-10-01 against each data set's own page or repository API. "direct" = no
account; "login" = free account; sizes as reported by the repository. `download_data.py --list`
prints the same list; it downloads the "direct" ones.

### Chest (far-field radar)

| Data set | Radar, rate | Subjects | BP reference | Size | Licence | Access | Link |
|---|---|---|---|---|---|---|---|
| **Erlangen phase 1** (Schellenberger 2020). Best chest set | 24 GHz CW, I/Q at 2000 Hz | 30 healthy, 24 h; rest, Valsalva, apnea, tilt | **continuous waveform** (Task Force Monitor) | 6.0 GB | CC BY 4.0 | direct | [figshare 12186516](https://doi.org/10.6084/m9.figshare.12186516) · files [1](https://ndownloader.figshare.com/files/22515785) [2](https://ndownloader.figshare.com/files/22516136) [3](https://ndownloader.figshare.com/files/22521941) |
| **TRCCBP** (Sensors 2023) | 7.3 GHz IR-UWB, 20 frames/s | 36 in the paper; 20-person folder with rest / apnea / sport | cuff value per recording | 1.7 GB | **no licence file** | direct | [github.com/bupt-uwb/TRCCBP](https://github.com/bupt-uwb/TRCCBP) |
| **110-participant vital signs** (Parralejo 2026) | 60 GHz IWR6843, range FFT at **10 Hz** | 110 (22-76 y), ~6 h | **one** cuff reading per person | 245 MB | CC BY 4.0 | direct | [Zenodo 18599983](https://doi.org/10.5281/zenodo.16760683) · [zip](https://zenodo.org/api/records/18599983/files/db_records.zip/content) |
| **VitalSense 120 GHz**, healthy | 120 GHz FMCW, ~333 Hz | 24, 2 x 2 min | 2 cuff readings per person | 29 MB | CC BY 4.0 | login | [IEEE DataPort 10.21227/wq68-sv85](https://ieee-dataport.org/open-access/new-dataset-millimeter-wave-radar-vital-sensing-reference-signals) |
| VitalSense 120 GHz, cardiovascular inpatients | 120 GHz FMCW | 15 patients | not confirmed | ~87 GB | not stated (under construction) | private link | [PATIENT.md](https://github.com/Rc-W024/VS_DATASET/blob/main/PATIENT.md) |
| PolyPulse example (Zhu 2026) | 77-81 GHz, heart / carotid / mastoid / **wrist** beams | 1 user | none | 5.75 GB | CC BY 4.0 | direct | [Zenodo 19403782](https://doi.org/10.5281/zenodo.19403782) |

Usable for BP training today: Erlangen (waveform level) and TRCCBP (cuff level, 20 Hz, licence
unclear). The 110-participant set has one label per person and a 10 Hz rate: too slow for pulse
shape, usable only for between-person checks.

Chest radar **without BP** (pre-training of the signal encoder only):

| Data set | Radar | Subjects | Other signals | Size | Licence | Access | Link |
|---|---|---|---|---|---|---|---|
| Erlangen heart sounds (Shi 2020) | 24 GHz CW, 2000 Hz | 11 | ECG, PCG | 584 MB | CC BY 4.0 | direct | [figshare 9691544](https://doi.org/10.6084/m9.figshare.9691544) |
| MMECG (Chen 2022) | 77 GHz AWR1843, 200 Hz | 35 | ECG | - | signed agreement | request | [github.com/jinbochen0823/RCG2ECG](https://github.com/jinbochen0823/RCG2ECG) |
| EquiPleth (Vilesov 2022) | 77 GHz AWR1443 + camera | 91 | PPG | - | not stated | request form | [github.com/UCLA-VMG/EquiPleth](https://github.com/UCLA-VMG/EquiPleth) |
| PhysDrive (2025) | 77 GHz, in car | 48 | ECG, PPG, respiration | - | academic use | Kaggle login | [github.com/WJULYW/PhysDrive-Dataset](https://github.com/WJULYW/PhysDrive-Dataset) |
| mmWave emotion (2026) | mmWave | 15 | PPG, GSR | 293 MB processed | CC BY 4.0 | direct | [Zenodo 16900785](https://doi.org/10.5281/zenodo.16900785) |
| Dual-subject HR / RR | IWR6843ISK + DCA1000 raw ADC | 20 | HR belt, respiration | - | CC BY 4.0 | direct | [figshare 33286755](https://doi.org/10.6084/m9.figshare.33286755.v2) |
| Multi-target vital signs (2024) | IWR6843ISK + DCA1000 raw ADC, 20 Hz | 2 | ECG | - | CC BY 4.0 | direct | [Mendeley 684v4r8wfr](https://data.mendeley.com/datasets/684v4r8wfr/1) |
| Child vital signs (Yoo 2021) | IWR6843, 20 frames/s | 50 children | HR, RR | 1.96 GB | CC0 | direct | [figshare 13515977](https://doi.org/10.6084/m9.figshare.13515977.v1) |

### Wrist (near-field radar, radial artery)

| Data set | Radar, rate | Subjects | BP reference | Size | Licence | Access | Link |
|---|---|---|---|---|---|---|---|
| **Blumio** wearable sensors (2021). Best wrist set | 60 GHz BGT60TR24B worn on the wrist; a 1-D radar waveform, no raw ADC | 115 (20-67 y), ~10 min each at rest | **continuous waveform** (CNAP 500) + PPG + tonometry | 112 MB | "Open Access" (text not shown) | login | [IEEE DataPort 10.21227/3yte-wz05](https://ieee-dataport.org/open-access/dataset-synchronized-signals-wearable-cardiovascular-monitoring-sensors) |
| **airBP** public sample (Liang 2023) | contact-free mmWave at the wrist, 210 Hz, 30 s clips | 15 clips of 5 subjects | cuff SBP / DBP per clip | 737 kB | MIT | direct | [github.com/YumengLiang/dataset-of-mmwave](https://github.com/YumengLiang/dataset-of-mmwave) · [zip](https://raw.githubusercontent.com/YumengLiang/dataset-of-mmwave/main/dataset_blood_pressure.zip) |
| PolyPulse example (wrist beam) | 77-81 GHz, stand-off | 1 user | none | see chest table | CC BY 4.0 | direct | [Zenodo 19403782](https://doi.org/10.5281/zenodo.19403782) |

That is all that is public for the wrist: one real training set (Blumio, contact wearable, at
rest only) and a 15-clip sample. There is no public stand-off wrist radar set with BP labels.

### No radar, with BP (pre-training or teacher models)

| Data set | Signals | Subjects | BP reference | Size | Licence | Access | Link |
|---|---|---|---|---|---|---|---|
| **Aurora-BP** (Microsoft). Closest to wrist radar | wrist **tonometry**, PPG, ECG | ~1200 | cuff, with BP-changing manoeuvres | 6.6 GB | Microsoft DUA (no redistribution, no merging with other data) | direct | [Zenodo 19099166](https://doi.org/10.5281/zenodo.19099166) |
| **Bed ballistocardiography** (Carlson 2021). Closest to chest radar | bed force sensors (contactless mechanical cardiac signal), ECG, PPG | 40 | continuous (Finometer) | 2.1 GB | CC BY per paper | login | [IEEE DataPort 10.21227/77hc-py84](https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset) |
| MIMIC-BP (Sanches 2024) | PPG, ECG, respiration | 1524 | invasive waveform | 0.9 GB | ODbL 1.0 | direct | [Harvard Dataverse DBM1NF](https://doi.org/10.7910/DVN/DBM1NF) |
| PulseDB v2 | PPG, ECG, 10 s segments | 5361 | invasive waveform | large | not stated | direct | [github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB) |
| VitalDB | ECG, PPG, 196 parameters | 6388 surgeries | invasive waveform | 95 GB | CC BY 4.0 | direct | [PhysioNet vitaldb](https://physionet.org/content/vitaldb/1.0.0/) |
| MIMIC-III PPG windows (Schrumpf) | PPG 7 s windows | 905,400 windows | SBP / DBP per window | 32 GB | CC BY 4.0 | direct | [Zenodo 5590603](https://doi.org/10.5281/zenodo.5590603) |
| Pulse Transit Time PPG | PPG, ECG, IMU; sit / walk / run | 22 | cuff before and after | 2.9 GB | ODbL 1.0 | direct | [PhysioNet pulse-transit-time-ppg](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) |
| UCI cuff-less BP | PPG, ECG | 12,000 records, no subject ids | invasive waveform | 3.1 GB | CC BY 4.0 | direct | [UCI 340](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation) |
| PPG-BP (Liang 2018) | fingertip PPG, 2.1 s | 219 | one cuff reading | 1.5 MB | CC0 | direct | [figshare 5459299](https://doi.org/10.6084/m9.figshare.5459299) |
| Pulse Wave Database (Charlton 2019) | **simulated** radial / carotid / aortic waves | 4374 virtual | exact | 243 MB (csv) | ODC PDDL | direct | [Zenodo 3275625](https://doi.org/10.5281/zenodo.3275625) |

These have no loader in this repo yet; they are listed for encoder pre-training.

### Not public

The data of WaveBP, mmBP, hBP-Fi, RF-BP, BP3, the full airBP set (41 subjects), the full
PolyPulse set, Erlangen "phase 2" and several smaller studies were not released (no repository
found, or the paper says "on request" / "privacy").

## Layout

```
train.py              cross-validated training, report, final model
download_data.py      direct downloads
selftest.py           installation check on synthetic data
mmwave_bp/
  data.py             Recording, loaders (erlangen, blumio, airbp, npz), labels, subject folds
  dsp.py              filters, I/Q ellipse correction, displacement, VMD, AMPD, beat tools
  engine.py           windows, training with resume, evaluation against baselines
  metrics.py          MAE / ME / SD, AAMI and ISO 81060-2 criteria, BHS, tracking
  methods/            wavebp.py  rfbp.py  airbp.py  mmbp.py  hbpfi.py
```

## Limits

Research code. A model trained here is not a medical device and its output is not a diagnosis;
report it together with the error measured on unseen people (`report.json`).
