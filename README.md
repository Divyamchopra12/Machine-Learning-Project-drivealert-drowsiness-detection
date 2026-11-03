# DriveAlert – Driver Drowsiness & Distraction Detection System

Real-time driver monitoring from a webcam using **MediaPipe facial landmarks**
(Eye Aspect Ratio, Mouth Aspect Ratio, head pose) fused with **MobileNetV2 CNNs**
trained on the Kaggle **Yawn-Eye Dataset**, with a full web dashboard.

## Features

| Area | Feature |
|---|---|
| Detection | Eye closure / microsleep (EAR + CNN fusion), blinks, PERCLOS, yawning (MAR + CNN), head nodding (pitch) |
| Distraction | Looking away from the road (head yaw) or face lost for > 2.5 s – inspired by the EU ADDW requirement |
| Alarm | **3-level escalation**: L1 early warning (chime + banner), L2 alarm (beeps + overlay), L3 critical (siren) after 3 s or 3 alarms in 10 min |
| Voice | Spoken alerts that get more urgent per level; voice commands (“I'm okay”, “find rest stop”, “recalibrate”, “stop monitoring”) |
| Emergency | Telegram message with live Google Maps location to a family member / fleet manager after repeated alarms |
| Rest stops | Nearby fuel stations, cafés, restaurants, rest areas from OpenStreetMap (Overpass API) on a Leaflet map, with navigation links |
| Trips | Every session saved: fatigue-per-minute chart, alarms, KSS self-rating, location, risk level, recommendations, **AI summary (Gemini, optional)**, Save as PDF |
| Sleepiness scale | Karolinska Sleepiness Scale (KSS 1–9) asked at trip start and end; estimated KSS shown live |
| Night mode | CLAHE low-light enhancement when the frame is dark |
| Total darkness | IR camera detected automatically → **night eye model** trained on the infrared MRL Eye Dataset; with a normal webcam and no light: "too dark" alert (no false distraction alarms), **voice check-ins** ("Are you awake?") with escalation and emergency alert after 2 misses, and gyroscope **swerve detection** |
| Breaks | Spoken break reminder every N minutes of driving |
| Fatigue forecast | Linear trend of the fatigue score warns “likely drowsy in X min” before a microsleep |
| Time-of-day & time-on-task | ×1.25 sensitivity at 1–5 AM, ×1.1 at 2–4 PM and 6 PM–1 AM; extra fatigue after 90 min of driving |
| Speed gating | No alarms below a set GPS speed (parked) |
| Sunglasses fallback | Hidden eyes detected → eye signals ignored, head pose + yawning used instead |
| Adaptive learning | “False alarm” button makes that alarm slightly harder to trigger for this driver (saved per driver) |
| Crash detection | Phone motion sensor > 4 g + no answer in 30 s → emergency alert with location |
| Research & Impact page | Official statistics (WHO, MoRTH, AIIMS, NHTSA, AAA, IIT Bombay, EU/India rules) each linked to the feature that answers it |
| Classical ML | 11 syllabus models (LR, KNN, SVM, DT, RF, NB, GB, AdaBoost, Extra Trees, MLP, LDA, + XGBoost) compared with the CNNs; best one selectable live |
| Deep learning | 5 CNN architectures compared: Custom CNN, MobileNetV2, EfficientNetB0, ResNet50V2, DenseNet121 |
| Temporal models | LSTM, GRU, BiLSTM and 1D-CNN on 3-second sequences of 8 signals (UTA-RLDD), blended into the live fatigue score |
| Phone use | EfficientDet-Lite0 (or YOLOv8n) detects a phone near the face → "Put the phone down" alarm |
| Fleet Manager | Per-driver trips, hours today vs daily duty limit, alarms/hour, safety-score leaderboard, CSV, plain-English questions (Gemini) |
| Languages | Voice alerts and commands in English, Hindi and Punjabi |
| User study | Built-in System Usability Scale (SUS) survey with scoring, grade and CSV export |
| Mobile app | Installable PWA (manifest + service worker) |
| Demo recorder | Records the screen + microphone to a .webm demo video |
| Deployment | Dockerfile + render.yaml for cloud; Raspberry Pi unit with buzzer, vibration motor, LEDs and MQ-3 check |
| Research | Ablation study (EAR vs CNN vs fusion vs classical), Grad-CAM explainability, TFLite int8 edge export |

## Project structure

```
drowsiness-detection/
├── train_model.py       # trains eye (open/closed) + yawn CNNs on Kaggle data
├── train_dl_compare.py  # 5 CNN architectures compared (Custom CNN, MobileNetV2, EfficientNetB0, ResNet50V2, DenseNet121)
├── train_classical.py   # 11 classical ML models vs the CNNs
├── features.py          # HOG + histogram features used by the classical models
├── train_night.py       # infrared (night) eye model on the MRL Eye Dataset + day-vs-night comparison
├── train_temporal.py    # LSTM / GRU / BiLSTM / 1D-CNN on 3-s landmark sequences (UTA-RLDD or own clips)
├── temporal.py          # shared sequence helpers (used live by detector.py)
├── phone_detector.py    # phone-in-hand detection (EfficientDet-Lite0 or YOLOv8n)
├── kaggle_train.ipynb   # ready-to-run Kaggle notebook (GPU)
├── detector.py          # core logic: EAR, MAR, head pose, PERCLOS, distraction, alarm levels, night mode
├── app.py               # FastAPI backend: WebSocket, trips, emergency alerts, AI reports, settings
├── run_webcam.py        # desktop app with OpenCV window + escalating alarm sound
├── evaluate.py          # ablation study on your own labelled videos
├── tune_fusion.py       # tunes fusion weights + alarm delay on labelled clips
├── significance.py      # bootstrap CIs + McNemar's test for all trained models
├── failure_gallery.py   # Grad-CAM gallery of mistakes + cause statistics
├── tests/               # pytest tests (run: pytest -q)
├── MODEL_CARD.md        # intended use, data, limitations, privacy
├── gradcam.py           # Grad-CAM heatmaps (where the CNN looks)
├── export_tflite.py     # TFLite fp32 / fp16 / int8 export + size/speed/accuracy benchmark
├── frontend/            # DriveAlert web UI (index.html, PWA manifest + sw.js, icons)
├── raspberry_pi/        # run_pi.py + wiring guide for the in-car unit
├── Dockerfile, render.yaml  # cloud deployment
├── models/              # eye_model.keras / yawn_model.keras (+ metrics, plots, .tflite)
├── data/                # saved events, trips and settings (created automatically)
└── requirements.txt
```

## 1. Train the models on Kaggle (free GPU)

1. New Kaggle notebook → File → Import Notebook → `kaggle_train.ipynb`.
2. **Add data** → `yawn-eye-dataset-new` (serenaraju); upload `train_model.py`, `train_dl_compare.py`,
   `train_classical.py` and `features.py`; turn **GPU** on; Run All.
3. Download `models_export.zip` from the Output tab and unzip into `models/`.

### Classical ML models (syllabus models)

```powershell
python train_classical.py --task both          # add --tune for GridSearchCV on RF and SVM
```

Trains **Logistic Regression, K-Nearest Neighbours, SVM (RBF), Decision Tree, Random Forest,
Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees, MLP (neural network) and LDA** on HOG features of the same images and the same test split as
the CNN. Reports accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation, training time,
inference time and model size (`results/<task>_classical_comparison.md`, comparison chart and a
Random Forest feature-importance map). The best model is saved as `models/<task>_classical.joblib`
and can be used live: **Settings → Eye-state model → Best classical ML + EAR**.
The Kaggle notebook runs this automatically after the CNN. Install `xgboost` to add XGBoost.

### Deep learning architectures

```powershell
python train_dl_compare.py --task both                      # all 5 CNNs (use a GPU – Kaggle)
python train_dl_compare.py --task eye --archs custom_cnn mobilenetv2 efficientnetb0
python train_dl_compare.py --task both --promote-best       # copy the best CNN to <task>_model.keras
```

Writes `models/<task>_dl_metrics.json`, `<task>_dl_comparison.png` and `results/<task>_dl_comparison.md`
(accuracy, F1, AUC, parameters, size, ms/image, training time). Shown on the About page.

### Temporal sequence models (LSTM / GRU)

Single images cannot show *slow* blinks or a head that drifts down over seconds. These models read
3-second windows (30 frames × 8 signals) extracted by `detector.py` from videos.

```powershell
# UTA Real-Life Drowsiness Dataset (folders with 0 = alert, 10 = drowsy videos)
python train_temporal.py --rldd D:\UTA-RLDD --max-seconds 120
# or your own clips: clips\labels.csv with columns file,label (alert / drowsy)
python train_temporal.py --videos clips
```

Trains LSTM, GRU, BiLSTM, 1D-CNN and a Random Forest baseline, split **by video**. The best one is
saved as `models/temporal_model.keras` and blended live into the fatigue score (0.7 × rules + 0.3 × model).
`--synthetic 400` only tests the code; such a model is never used live.

### Night / infrared eye model

A normal webcam cannot see in a completely dark car. Real systems use an **infrared camera with
850/940 nm IR LEDs** (invisible to the driver). Their images are grey, so a second eye CNN is trained
on infrared images:

```powershell
# MRL Eye Dataset: http://mrl.cs.vsb.cz/eyedataset (or a Kaggle copy – search "MRL eye dataset")
python train_night.py --data-dir D:\mrlEyes_2018_01 --by-subject
```

Saves `models/eye_night.keras` and `results/night_vs_day.md` (day model vs night model on the same
infrared test images). The app switches to it automatically when the camera image is infrared or dark.

Without an IR camera, in total darkness the app shows **TOO DARK** (instead of a false "eyes on road"
alarm), asks **"Are you awake?"** by voice every few minutes (answer by voice or button; no answer →
alarm → siren → emergency contact after 2 misses) and uses the phone gyroscope to detect swerving.
Settings: *Night voice check-in*, *Swerve detection*, *"Too dark" level*, *Night (infrared) eye model*.
Demo: start monitoring and cover the webcam with your hand.

### Phone-use detection

Turn on **Settings → Phone-use detection**. The first run downloads EfficientDet-Lite0 (~4 MB, COCO
"cell phone" class) into `models/`. For YOLOv8n instead: `pip install ultralytics`.

## 2. Install and run (Windows, Python 3.12)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --port 8000      # open http://localhost:8000
```

Desktop-only version: `python run_webcam.py` (q = quit, r = recalibrate).

## 3. Optional setup (Settings page)

- **Emergency alert:** Telegram → @BotFather → `/newbot` → copy the bot token. The contact sends any
  message to the bot, then gets their chat ID from @userinfobot. Paste both, enable, press *Send test message*.
- **AI trip summaries:** free API key from aistudio.google.com → paste into *Gemini API key*
  (or set the `GEMINI_API_KEY` environment variable). Without it, rule-based reports still work.
- Secrets are stored only in `data/settings.json` on your computer and never sent back to the browser.

## 4. New pages

- **Fleet Manager** – every driver (the *Driver name* in Settings) with trips, hours today vs the
  *daily duty limit*, alarms per hour and a safety score. "Drive as" switches driver. Ask questions in
  plain English with a Gemini key.
- **User Study (SUS)** – let 10+ people try the app, each fills the 10-question survey; mean score,
  grade and CSV export for the report. Average SUS is 68.
- **REC DEMO** (top bar) – records the screen + your voice; the .webm downloads when you press STOP.
- **INSTALL APP** (top bar, Chrome/Edge) – installs DriveAlert like a phone app.
- **Voice language** (Settings) – English, Hindi or Punjabi alerts. Windows/Android need the
  matching speech voice installed, otherwise the default voice is used.

## 5. Deploy

```bash
docker build -t drivealert .
docker run -p 8000:8000 drivealert                 # http://localhost:8000
```

On Render: push the project (with `models/`) to GitHub → render.com → New → Blueprint. Cameras only
work on `localhost` or **HTTPS** (Render gives HTTPS). The free plan has 512 MB RAM, which may not be
enough for TensorFlow + CNNs; switch Eye-state model to *EAR only* or use a paid instance.
Raspberry Pi: see `raspberry_pi/README.md`.

## 6. Research rigour (what examiners look for)

```powershell
# 1. Is model A really better than B?  95% bootstrap CIs + McNemar's test on the same test images
python significance.py --task eye            # -> results/eye_significance.md (About page)

# 2. Where and why the CNN fails: most confident mistakes + Grad-CAM + likely cause
python failure_gallery.py --task eye         # -> results/eye_failures.md / .png (About page)

# 3. Tune fusion weight, threshold and alarm delay on your own labelled clips (labels as for evaluate.py)
python tune_fusion.py --videos clips --people amber ravi      # tune on some people …
python evaluate.py   --videos clips_test                       # … report on OTHER people
python tune_fusion.py --videos clips --apply                   # save the tuned values for the app
```

Also included: `MODEL_CARD.md` (intended use, data, limits, privacy – fill the numbers), automated
tests (`pytest -q`, 19 tests) and a GitHub Actions workflow (`.github/workflows/tests.yml`) that runs
them on every push.

## 7. Research experiments (for the report)

```powershell
# Ablation study on your own clips (EAR-only vs CNN-only vs fusion)
python evaluate.py --videos clips --make-template   # then edit clips/labels.csv
python evaluate.py --videos clips                   # -> results/ablation.md, conditions.md, events.md

# Explainability
python gradcam.py --model models/eye_model.keras --images <dataset>/train/Closed <dataset>/train/Open

# Edge deployment (Raspberry Pi / phone)
python export_tflite.py --data-dir <dataset>        # -> results/tflite_benchmark.md
```

## How detection works

| Signal | Method | Triggers |
|---|---|---|
| Eye closure | EAR (6 landmarks per eye) fused with eye CNN: closed if 0.6·P(closed) + 0.4·[EAR < thr] ≥ 0.5 | ≥ 1.5 s → alarm |
| PERCLOS | % of frames with eyes closed in the last 60 s | fatigue score |
| Yawning | MAR > 0.6 for ≥ 1 s (or yawn CNN > 0.85) | fatigue score, warning |
| Head nod | solvePnP pitch vs calibrated neutral | > 18° for 2 s → alarm |
| Distraction | solvePnP yaw vs neutral, or face lost | > 30° / lost for 2.5 s → alarm |

**Fatigue score (0–100)** = 50 × PERCLOS/30% + 25 × yawns(5 min)/3 + 25 × nods(5 min)/3, each part capped;
an ongoing eye closure pushes it towards 100. **Estimated KSS** = 1 + score/100 × 8 (approximation, not validated).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | status, models, AI / emergency readiness |
| GET / PUT | `/api/config` | detection thresholds (saved) |
| GET / PUT | `/api/settings` | driver name, emergency contact, Gemini (secrets never returned) |
| POST | `/api/notify/test` | send a test Telegram message |
| GET / DELETE | `/api/events` | alarm (critical), warning and KSS (info) log |
| GET / DELETE | `/api/trips` | saved trips |
| GET | `/api/trips/{session}` | one trip with fatigue timeline |
| POST | `/api/trips/{session}/report` | `{"use_ai": true}` → rule-based + AI report |
| GET | `/api/model-info`, `/api/model-plot/{name}` | training results (CNN, DL comparison, classical, temporal) |
| GET | `/api/fleet` | per-driver totals, hours today, safety score, drivers over the duty limit |
| POST | `/api/fleet/ask` | `{"question": "..."}` → Gemini answer from fleet data |
| GET / POST / DELETE | `/api/sus` | SUS questions, responses and summary / submit `{"answers": [1..5 ×10]}` |
| WS | `/ws/detect` | send `frame`, `recalibrate`, `ack`, `location`, `kss`; receive per-frame results and `emergency` notices |

Per-frame results include `status` (`alert`, `warning`, `drowsy`, `distracted`, `calibrating`, `no_face`),
`alarm`, `alarm_reason`, `alarm_level` (0–3), `fatigue_score`, `est_kss`, `ear`, `mar`, `perclos`, `blink_rate`,
`yawn_count`, `head_pitch`, `head_yaw`, `looking_away_seconds`, `night_mode`, `brightness` and `latency_ms`.

## Notes and limitations

- The Yawn-Eye dataset has few subjects; the random split can share people between train and test.
  Use `evaluate.py` on your own recordings for an honest real-world figure.
- Sunglasses block eye detection; an IR camera works better at night than CLAHE enhancement.
- Voice commands need Chrome or Edge; location and camera need `localhost` or HTTPS.
- This is a student/research prototype, not a certified automotive safety system.
