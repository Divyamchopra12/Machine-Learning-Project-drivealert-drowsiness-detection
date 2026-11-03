"""
FastAPI backend for the DriveAlert web frontend.

  python -m uvicorn app:app --port 8000      ->  open http://localhost:8000

The browser captures webcam frames and sends them over a WebSocket; the
server runs the detector and replies with metrics for each frame.

Endpoints
  GET    /api/health               server + model status
  GET    /api/config               detection thresholds
  PUT    /api/config               update thresholds (applies to live sessions, saved)
  GET    /api/settings             app settings (driver, emergency alert, AI report)
  PUT    /api/settings             update settings (saved)
  POST   /api/notify/test          send a test Telegram emergency message
  GET    /api/events               alarm / warning / self-report log, newest first
  DELETE /api/events               clear the log
  GET    /api/trips                finished trips (summaries)
  GET    /api/trips/{session}      one trip with its fatigue timeline
  POST   /api/trips/{session}/report   trip report (rule-based, or AI with Gemini)
  DELETE /api/trips                clear trip history
  GET    /api/model-info           training results for the About page
  GET    /api/model-plot/NAME      training plots (PNG)
  WS     /ws/detect                {"type":"frame","image":...} | {"type":"recalibrate"} |
                                   {"type":"ack"} | {"type":"location","lat":..,"lon":..} |
                                   {"type":"kss","value":1-9,"phase":"start"|"end"}
  GET    /                         serves ./frontend

Data (events, trips, settings) is stored as JSON in ./data so it survives restarts.
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, deque
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

mimetypes.add_type("application/manifest+json", ".webmanifest")
from pydantic import BaseModel

from detector import (KSS_LABELS, DetectorConfig, DrowsinessDetector, load_classical_model,
                      load_keras_model)
from phone_detector import get_phone_detector
from temporal import load_temporal

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
EVENTS_FILE, TRIPS_FILE, SETTINGS_FILE = (DATA_DIR / f for f in
                                          ("events.json", "trips.json", "settings.json"))
EYE_MODEL = str(MODEL_DIR / "eye_model.keras")
YAWN_MODEL = str(MODEL_DIR / "yawn_model.keras")
PLOTS = {f"{t}_{k}.png" for t in ("eye", "yawn")
         for k in ("confusion_matrix", "training_curves", "model_comparison", "rf_importance", "dl_comparison",
                   "failures")} | {"fusion_tuning.png"}
EYE_MODES = ("fusion", "ear", "cnn", "classical")

CRITICAL = {"eyes_closed", "head_down", "distracted", "phone_use", "high_fatigue", "crash", "checkin_missed"}
NIGHT_EVENTS = {"swerving": "warning", "checkin_missed": "critical", "checkin_ok": "info"}
SUS_FILE = DATA_DIR / "sus.json"
WARNING_COOLDOWN = 15.0  # seconds between repeated "rising_fatigue" warnings
MIN_TRIP_SECONDS = 10


# ------------------------------------------------------------ storage ----
def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


DEFAULT_SETTINGS = {
    "driver_name": "Driver",
    "emergency_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "emergency_alarm_count": 3,      # this many alarms ...
    "emergency_window_min": 10,      # ... within this many minutes -> alert contact
    "gemini_api_key": "",            # optional, or set the GEMINI_API_KEY env variable
    "gemini_model": "gemini-2.5-flash",
    "adaptive_learning": True,       # learn per-driver thresholds from "false alarm" feedback
    "duty_limit_hours": 8.0,         # warn when a driver exceeds this much driving in a day
    "profiles": {},                  # driver name -> learned thresholds
}
SECRET_KEYS = ("telegram_bot_token", "gemini_api_key")

SETTINGS = {**DEFAULT_SETTINGS, **_load(SETTINGS_FILE, {})}
_saved_cfg = SETTINGS.pop("detector", {})
CONFIG = DetectorConfig(**{k: v for k, v in _saved_cfg.items()
                           if k in {f.name for f in fields(DetectorConfig)}})
EVENTS: deque = deque(_load(EVENTS_FILE, []), maxlen=1000)
TRIPS: list = _load(TRIPS_FILE, [])
_EVENT_IDS = itertools.count(max((e.get("id", 0) for e in EVENTS), default=0) + 1)
SESSIONS: set[DrowsinessDetector] = set()


def save_settings():
    _save(SETTINGS_FILE, {**SETTINGS, "detector": asdict(CONFIG)})


def save_events():
    _save(EVENTS_FILE, list(EVENTS))


# ------------------------------------------------------ notifications ----
def send_telegram(text: str) -> tuple[bool, str]:
    token, chat = SETTINGS["telegram_bot_token"].strip(), str(SETTINGS["telegram_chat_id"]).strip()
    if not token or not chat:
        return False, "Telegram bot token or chat ID is missing"
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat, "text": text, "disable_web_page_preview": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = json.loads(resp.read()).get("ok", False)
            return ok, "sent" if ok else "Telegram refused the message"
    except urllib.error.HTTPError as exc:
        return False, f"Telegram error {exc.code}: check the bot token and chat ID"
    except Exception as exc:  # network down etc.
        return False, f"Could not reach Telegram: {exc}"


def maps_link(loc):
    return f"https://maps.google.com/?q={loc['lat']:.6f},{loc['lon']:.6f}" if loc else None


# ---------------------------------------------------------- AI report ----
def rule_report(trip: dict) -> dict:
    """Plain-language trip report built from the numbers (no AI needed)."""
    mins, secs = divmod(int(trip["duration_s"]), 60)
    alarms, warns = trip["alarms"], trip["warnings"]
    n_alarms = sum(alarms.values())
    risk = ("High" if n_alarms >= 3 or trip["max_fatigue"] >= 90 else
            "Moderate" if n_alarms >= 1 or trip["avg_fatigue"] >= 35 else "Low")
    paras = [f"{trip['driver']} drove for {mins} min {secs} s. "
             f"Average fatigue score was {trip['avg_fatigue']:.0f}/100 (peak {trip['max_fatigue']:.0f}), "
             f"with eyes closed up to {trip['max_perclos'] * 100:.0f}% of the time in a one-minute window."]
    if n_alarms:
        parts = [f"{v} × {k.replace('_', ' ')}" for k, v in alarms.items()]
        paras.append(f"There were {n_alarms} alarm(s): {', '.join(parts)}. "
                     f"The longest eye closure lasted {trip['longest_closure']:.1f} s.")
    else:
        paras.append("No drowsiness alarms were triggered.")
    if warns:
        paras.append("Early warning signs: " + ", ".join(f"{v} × {k.replace('_', ' ')}"
                                                        for k, v in warns.items()) + ".")
    tl = trip.get("timeline") or []
    if len(tl) >= 3:
        first, last = np.mean([p[1] for p in tl[:len(tl) // 3]]), np.mean([p[1] for p in tl[-len(tl) // 3:]])
        if last > first + 10:
            paras.append(f"Fatigue rose during the trip (from about {first:.0f} to {last:.0f}), "
                         "which suggests the driver was getting tired as time went on.")
    if trip.get("kss_start") and trip.get("kss_end"):
        paras.append(f"Self-reported sleepiness (KSS) went from {trip['kss_start']} "
                     f"({KSS_LABELS[trip['kss_start']].lower()}) to {trip['kss_end']} "
                     f"({KSS_LABELS[trip['kss_end']].lower()}).")
    if trip.get("night_ratio", 0) > 0.3:
        paras.append(f"{trip['night_ratio'] * 100:.0f}% of the trip was in low light; night mode enhancement was used.")
    recs = []
    if risk == "High":
        recs.append("Do not continue driving now: take a 20-minute nap or switch drivers.")
    if n_alarms:
        recs.append("Plan a break every 45–60 minutes on long drives.")
    if warns.get("yawn", 0) >= 2:
        recs.append("Frequent yawning: ventilate the cabin, drink water and avoid heavy meals before driving.")
    if alarms.get("distracted"):
        recs.append("Keep your eyes on the road; set up navigation and music before starting.")
    if alarms.get("phone_use"):
        recs.append("Phone use was detected while driving: put the phone in a mount or on Do Not Disturb.")
    if warns.get("too_dark"):
        recs.append("The cabin was too dark for the camera: fit an infrared camera with IR LEDs, "
                    "or keep a dim cabin light on; voice check-ins were used meanwhile.")
    if alarms.get("checkin_missed"):
        recs.append(f"{alarms['checkin_missed']} night voice check-in(s) went unanswered – a strong sign of sleep. "
                    "Stop and rest before driving on.")
    if warns.get("swerving"):
        recs.append("Sudden steering corrections were detected by the motion sensor: slow down and take a break.")
    if alarms.get("crash"):
        paras.append(f"A possible crash or hard impact was detected {alarms['crash']} time(s); "
                     "the emergency contact was notified if alerts were enabled.")
    if trip.get("false_alarms"):
        paras.append(f"The driver marked {trip['false_alarms']} alarm(s) as false; the system adjusted "
                     "this driver's thresholds so similar situations are less likely to trigger again.")
    if trip.get("risk_period") and trip["risk_period"] != "Daytime":
        paras.append(f"Part of the trip fell in a higher-risk time window ({trip['risk_period'].lower()}), "
                     "when natural alertness is lower, so the system was more sensitive.")
    if (trip.get("kss_end") or 0) >= 7:
        recs.append("Your self-rated sleepiness was high (KSS ≥ 7). Sleep before the next drive.")
    recs.append("General guidance: take a 15-minute break at least every 2 hours of driving.")
    return {"source": "rules", "risk": risk, "paragraphs": paras, "recommendations": recs}


def gemini_report(trip: dict) -> dict:
    key = SETTINGS["gemini_api_key"].strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No Gemini API key configured")
    facts = {k: v for k, v in trip.items() if k not in ("timeline",)}
    facts["fatigue_timeline_per_minute"] = trip.get("timeline", [])
    prompt = (
        "You are a road-safety assistant. Write a short, friendly driver fatigue report "
        "(3 short paragraphs, plain text, no markdown) from this JSON trip data recorded by a "
        "webcam drowsiness detection system. Explain what happened, how risky it was, and give "
        "3 concrete recommendations. Do not invent numbers that are not in the data.\n\n"
        + json.dumps(facts, indent=1))
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{SETTINGS['gemini_model']}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return {"source": "ai", "model": SETTINGS["gemini_model"], "text": text}


# -------------------------------------------------------- session log ----
class SessionLog:
    """Turns the per-frame results of one session into log events and a trip summary.

    critical : an alarm (eyes_closed / head_down / distracted / high_fatigue). Updated
               while it lasts; when it ends we store its duration and whether the
               driver dismissed it ("acknowledged") or it cleared by itself.
    warning  : a yawn, or the status rising from alert to warning.
    info     : the driver's KSS self-rating.
    """

    def __init__(self):
        self.session = uuid.uuid4().hex[:6].upper()
        self.location = self.start_location = None
        self.kss_start = self.kss_end = None
        self.emergencies = 0
        self.last_emergency = float("-inf")
        self.false_alarms = 0
        self.risk_period = "Daytime"
        self.last_critical = None
        self.reset()

    def reset(self):
        self.active = None
        self.last_yawns = 0
        self.last_status = None
        self.last_warn = float("-inf")
        self.blind_logged = False
        self.missed_checkins = 0
        self.new_critical = False
        self.start = self.last_t = None
        self.frames = self.face_frames = self.night_frames = 0
        self.score_sum = self.max_score = self.max_perclos = self.longest_closure = 0.0
        self.minutes: dict[int, list] = {}

    def _new(self, r, severity, reason, **extra):
        ev = {
            "id": next(_EVENT_IDS), "session": self.session, "time": r.get("timestamp", time.time()),
            "severity": severity, "reason": reason,
            "fatigue_score": r.get("fatigue_score", 0), "perclos": r.get("perclos", 0),
            "ear": r.get("ear"), "eyes_closed_seconds": r.get("eyes_closed_seconds", 0),
            "head_pitch": r.get("head_pitch"), "head_yaw": r.get("head_yaw"),
            "looking_away_seconds": r.get("looking_away_seconds", 0),
            "yawn_count": r.get("yawn_count", 0), "alarm_level": r.get("alarm_level", 0),
            "location": self.location, "duration": None, "acked_after": None,
            "resolution": "active" if severity == "critical" else "logged", **extra,
        }
        EVENTS.append(ev)
        save_events()
        return ev

    def update(self, r):
        t = r["timestamp"]
        self.new_critical = False
        if r["calibrating"]:
            self.last_status = r["status"]
            return
        # trip statistics
        self.start = self.start or t
        self.last_t = t
        self.frames += 1
        self.night_frames += r["night_mode"]
        if r["face_detected"]:
            self.face_frames += 1
            s = r["fatigue_score"]
            self.score_sum += s
            self.max_score = max(self.max_score, s)
            self.max_perclos = max(self.max_perclos, r["perclos"])
            m = self.minutes.setdefault(int((t - self.start) // 60), [0.0, 0])
            m[0] += s
            m[1] += 1
        self.longest_closure = max(self.longest_closure, r["eyes_closed_seconds"])
        if r.get("circadian_factor", 1) > 1 and self.risk_period == "Daytime":
            self.risk_period = r["risk_period"]
        # critical alarms
        if r["alarm"]:
            if self.active is None:
                self.active = self._new(r, "critical", r["alarm_reason"])
                self.last_critical = self.active
                self.new_critical = True
            ev = self.active
            ev["fatigue_score"] = max(ev["fatigue_score"], r["fatigue_score"])
            ev["eyes_closed_seconds"] = max(ev["eyes_closed_seconds"], r["eyes_closed_seconds"])
            ev["looking_away_seconds"] = max(ev["looking_away_seconds"], r["looking_away_seconds"])
            ev["perclos"] = max(ev["perclos"], r["perclos"])
            ev["alarm_level"] = max(ev["alarm_level"], r["alarm_level"])
        elif self.active is not None:
            ev = self.active
            ev["duration"] = round(t - ev["time"], 1)
            ev["resolution"] = "acknowledged" if ev["acked_after"] is not None else "auto_resolved"
            self.active = None
            save_events()
        # warnings
        if r.get("camera_blind") and not self.blind_logged:
            self._new(r, "warning", "too_dark")      # once per dark episode
            self.blind_logged = True
        elif not r.get("camera_blind") and r["face_detected"]:
            self.blind_logged = False
        if r["yawn_count"] > self.last_yawns:
            self._new(r, "warning", "yawn")
        self.last_yawns = r["yawn_count"]
        if (r["status"] == "warning" and self.last_status == "alert" and not r["yawning"]
                and not r["looking_away"] and t - self.last_warn >= WARNING_COOLDOWN):
            self._new(r, "warning", "rising_fatigue")
            self.last_warn = t
        self.last_status = r["status"]

    def ack(self):
        if self.active is not None and self.active["acked_after"] is None:
            self.active["acked_after"] = round(time.time() - self.active["time"], 1)
            return True
        return False

    def mark_false_alarm(self):
        """Driver says the latest alarm (active or within the last 2 minutes) was wrong."""
        ev = self.last_critical
        if ev is None or ev.get("false_alarm") or ev["reason"] == "crash" or time.time() - ev["time"] > 120:
            return None
        ev["false_alarm"] = True
        if ev is self.active:
            ev["acked_after"] = ev["acked_after"] or round(time.time() - ev["time"], 1)
        self.false_alarms += 1
        save_events()
        return ev["reason"]

    def crash(self, g: float):
        self.new_critical = False
        return self._new({"timestamp": time.time(), "fatigue_score": 0}, "critical", "crash",
                         g_force=round(float(g), 1), resolution="emergency")

    def night_event(self, reason: str, detail=None):
        """Events from the browser's night fallback: missed voice check-in, swerving (motion sensor)."""
        r = {"timestamp": time.time(), "fatigue_score": 0}
        if reason == "checkin_missed":
            self.missed_checkins += 1
        elif reason == "checkin_ok":
            self.missed_checkins = 0
        extra = {"resolution": "logged"} if NIGHT_EVENTS[reason] == "critical" else {}
        if detail is not None:
            extra["detail"] = str(detail)[:80]
        return self._new(r, NIGHT_EVENTS[reason], reason, **extra)

    def checkin_text(self) -> str:
        link = maps_link(self.location)
        return "\n".join([
            f"⚠️ DriveAlert: {SETTINGS['driver_name']} did not answer {self.missed_checkins} voice check-ins in a row "
            "while driving in the dark (the camera could not see the driver).",
            f"Time: {time.strftime('%d %b %Y, %H:%M:%S')}",
            f"Location: {link}" if link else "Location: not shared",
            "Please call the driver and ask them to stop and rest."])

    def crash_text(self, g: float) -> str:
        link = maps_link(self.location)
        return "\n".join([
            f"🚨 DriveAlert CRASH ALERT: a hard impact (~{g:.1f} g) was detected for {SETTINGS['driver_name']} "
            "and the driver did not respond within 30 seconds.",
            f"Time: {time.strftime('%d %b %Y, %H:%M:%S')}",
            f"Location: {link}" if link else "Location: not shared",
            "Please call the driver immediately. If unreachable, call emergency services (112 in India)."])

    def set_location(self, lat, lon, acc=None):
        self.location = {"lat": round(float(lat), 6), "lon": round(float(lon), 6),
                         "accuracy": acc and round(float(acc))}
        self.start_location = self.start_location or self.location

    def set_kss(self, value, phase):
        value = int(value)
        if value not in KSS_LABELS:
            raise ValueError("KSS must be 1-9")
        if phase == "end":
            self.kss_end = value
        else:
            self.kss_start = value
        self._new({"timestamp": time.time()}, "info", "kss_report", kss=value, phase=phase)

    def emergency_due(self) -> bool:
        if not (self.new_critical and SETTINGS["emergency_enabled"]):
            return False
        window = SETTINGS["emergency_window_min"] * 60
        now = time.time()
        recent = [e for e in EVENTS if e["session"] == self.session and e["severity"] == "critical"
                  and now - e["time"] <= window]
        if len(recent) >= SETTINGS["emergency_alarm_count"] and now - self.last_emergency > window:
            self.last_emergency = now
            self.emergencies += 1
            return True
        return False

    def emergency_text(self, r) -> str:
        n = SETTINGS["emergency_alarm_count"]
        lines = [f"⚠️ DriveAlert emergency: {SETTINGS['driver_name']} may be falling asleep at the wheel.",
                 f"{n}+ drowsiness alarms in the last {SETTINGS['emergency_window_min']} minutes "
                 f"(latest: {r['alarm_reason'].replace('_', ' ')}, fatigue {r['fatigue_score']:.0f}/100).",
                 f"Time: {time.strftime('%d %b %Y, %H:%M:%S')}"]
        link = maps_link(self.location)
        lines.append(f"Location: {link}" if link else "Location: not shared")
        lines.append("Please call the driver and ask them to stop and rest.")
        return "\n".join(lines)

    def close(self):
        if self.active is not None:
            self.active["duration"] = round(time.time() - self.active["time"], 1)
            self.active["resolution"] = "session_ended"
            self.active = None
            save_events()
        if self.start is None or self.last_t - self.start < MIN_TRIP_SECONDS:
            return None
        evs = [e for e in EVENTS if e["session"] == self.session]
        trip = {
            "session": self.session, "driver": SETTINGS["driver_name"],
            "start": self.start, "end": self.last_t, "duration_s": round(self.last_t - self.start, 1),
            "frames": self.frames, "face_ratio": round(self.face_frames / max(self.frames, 1), 3),
            "avg_fatigue": round(self.score_sum / max(self.face_frames, 1), 1),
            "max_fatigue": round(self.max_score, 1), "max_perclos": round(self.max_perclos, 3),
            "longest_closure": round(self.longest_closure, 2),
            "alarms": dict(Counter(e["reason"] for e in evs if e["severity"] == "critical")),
            "warnings": dict(Counter(e["reason"] for e in evs if e["severity"] == "warning")),
            "kss_start": self.kss_start, "kss_end": self.kss_end,
            "start_location": self.start_location, "end_location": self.location,
            "night_ratio": round(self.night_frames / max(self.frames, 1), 3),
            "emergency_alerts": self.emergencies,
            "false_alarms": self.false_alarms,
            "risk_period": self.risk_period,
            "timeline": [[m, round(s / n, 1)] for m, (s, n) in sorted(self.minutes.items())],
        }
        TRIPS.append(trip)
        _save(TRIPS_FILE, TRIPS)
        return trip


# ------------------------------------------------------------ REST API ----
app = FastAPI(title="DriveAlert – Driver Drowsiness Detection API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ConfigUpdate(BaseModel):
    ear_threshold: float | None = None
    mar_threshold: float | None = None
    eye_closed_seconds: float | None = None
    yawn_seconds: float | None = None
    head_tilt_degrees: float | None = None
    head_down_seconds: float | None = None
    distraction_degrees: float | None = None
    distraction_seconds: float | None = None
    perclos_window: float | None = None
    alarm_score: float | None = None
    low_light_enhance: bool | None = None
    low_light_threshold: float | None = None
    calibrate: bool | None = None
    calibration_seconds: float | None = None
    circadian_risk: bool | None = None
    time_on_task_min: float | None = None
    speed_gate_kmh: float | None = None
    eye_occlusion_check: bool | None = None
    eye_mode: str | None = None
    use_temporal: bool | None = None
    phone_detection: bool | None = None
    phone_seconds: float | None = None
    dark_threshold: float | None = None
    fusion_weight: float | None = None
    fusion_threshold: float | None = None
    night_eye_model: bool | None = None


class SettingsUpdate(BaseModel):
    driver_name: str | None = None
    emergency_enabled: bool | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    emergency_alarm_count: int | None = None
    emergency_window_min: int | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    adaptive_learning: bool | None = None
    duty_limit_hours: float | None = None


class ReportRequest(BaseModel):
    use_ai: bool = False


@app.on_event("startup")
def warm_up():
    load_keras_model(EYE_MODEL)
    load_keras_model(YAWN_MODEL)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {"eye": load_keras_model(EYE_MODEL) is not None,
                   "yawn": load_keras_model(YAWN_MODEL) is not None,
                   "night": (MODEL_DIR / "eye_night.keras").exists(),
                   "classical": (load_classical_model(str(MODEL_DIR / "eye_classical.joblib")) or {}).get("name")},
        "eye_mode": CONFIG.eye_mode,
        "temporal": (load_temporal(str(MODEL_DIR)) or {}).get("meta", {}).get("best_model"),
        "phone": get_phone_detector(str(MODEL_DIR)).backend if CONFIG.phone_detection else None,
        "active_sessions": len(SESSIONS),
        "ai_report": bool(SETTINGS["gemini_api_key"] or os.environ.get("GEMINI_API_KEY")),
        "emergency_ready": bool(SETTINGS["emergency_enabled"] and SETTINGS["telegram_bot_token"]
                                and SETTINGS["telegram_chat_id"]),
    }


@app.get("/api/config")
def get_config():
    return asdict(CONFIG)


@app.put("/api/config")
def update_config(update: ConfigUpdate):
    valid = {f.name for f in fields(DetectorConfig)}
    changes = {k: v for k, v in update.model_dump().items() if v is not None and k in valid}
    if changes.get("eye_mode") not in (None, *EYE_MODES):
        raise HTTPException(400, f"eye_mode must be one of {EYE_MODES}")
    for k, v in changes.items():
        setattr(CONFIG, k, v)
    for det in SESSIONS:  # live sessions pick up new thresholds
        for k, v in changes.items():
            setattr(det.cfg, k, v)
    save_settings()
    return asdict(CONFIG)


def public_settings():
    out = {k: v for k, v in SETTINGS.items() if k not in SECRET_KEYS and k != "profiles"}
    out["learned_profile"] = SETTINGS["profiles"].get(SETTINGS["driver_name"])
    for k in SECRET_KEYS:
        out[k + "_set"] = bool(SETTINGS[k])
    out["gemini_env_key"] = bool(os.environ.get("GEMINI_API_KEY"))
    return out


@app.get("/api/settings")
def get_settings():
    return public_settings()


@app.put("/api/settings")
def update_settings(update: SettingsUpdate):
    for k, v in update.model_dump().items():
        if v is None:
            continue
        if k in SECRET_KEYS and v == "":  # empty = keep the stored secret
            continue
        SETTINGS[k] = v.strip() if isinstance(v, str) else v
    SETTINGS["emergency_alarm_count"] = max(1, int(SETTINGS["emergency_alarm_count"]))
    SETTINGS["emergency_window_min"] = max(1, int(SETTINGS["emergency_window_min"]))
    SETTINGS["duty_limit_hours"] = min(24.0, max(0.5, float(SETTINGS["duty_limit_hours"])))
    save_settings()
    return public_settings()


@app.delete("/api/settings/profile")
def reset_profile():
    """Forget what was learned from false-alarm feedback for the current driver."""
    SETTINGS["profiles"].pop(SETTINGS["driver_name"], None)
    save_settings()
    return public_settings()


@app.delete("/api/settings/secret/{name}")
def clear_secret(name: str):
    if name not in SECRET_KEYS:
        raise HTTPException(404, "unknown secret")
    SETTINGS[name] = ""
    save_settings()
    return public_settings()


@app.post("/api/notify/test")
async def notify_test():
    ok, msg = await asyncio.to_thread(
        send_telegram, f"✅ DriveAlert test message for {SETTINGS['driver_name']}. "
                       "Emergency alerts are set up correctly.")
    return {"ok": ok, "message": msg}


@app.get("/api/events")
def get_events(limit: int = 500):
    return list(EVENTS)[-limit:][::-1]


@app.delete("/api/events")
def clear_events():
    EVENTS.clear()
    save_events()
    return {"cleared": True}


@app.get("/api/trips")
def get_trips():
    return [{k: v for k, v in t.items() if k != "timeline"} for t in reversed(TRIPS)]


def _find_trip(session):
    for t in TRIPS:
        if t["session"] == session:
            return t
    raise HTTPException(404, "trip not found")


@app.get("/api/trips/{session}")
def get_trip(session: str):
    return _find_trip(session)


@app.post("/api/trips/{session}/report")
async def trip_report(session: str, req: ReportRequest):
    trip = _find_trip(session)
    report = rule_report(trip)
    if req.use_ai:
        try:
            report["ai"] = await asyncio.to_thread(gemini_report, trip)
        except Exception as exc:
            report["ai_error"] = f"AI report unavailable ({exc}); showing the rule-based report."
    return report


@app.delete("/api/trips")
def clear_trips():
    TRIPS.clear()
    _save(TRIPS_FILE, TRIPS)
    return {"cleared": True}


# ---------------------------------------------------------------- fleet ----
def _day(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def duty_hours_today(driver: str) -> float:
    today = _day(time.time())
    return round(sum(t["duration_s"] for t in TRIPS if t["driver"] == driver and _day(t["start"]) == today) / 3600, 2)


def fleet_summary() -> dict:
    drivers = {}
    names = {t["driver"] for t in TRIPS} | set(SETTINGS["profiles"]) | {SETTINGS["driver_name"]}
    for name in sorted(names):
        trips = [t for t in TRIPS if t["driver"] == name]
        secs = sum(t["duration_s"] for t in trips)
        hours = secs / 3600
        alarms = sum(sum(t["alarms"].values()) for t in trips)
        warns = sum(sum(t["warnings"].values()) for t in trips)
        avg = sum(t["avg_fatigue"] * t["duration_s"] for t in trips) / secs if secs else 0.0
        kss = [t["kss_end"] for t in trips if t.get("kss_end")]
        per_hr = alarms / hours if hours else 0.0
        # simple 0-100 safety score: fewer alarms per hour and lower fatigue = safer
        safety = max(0.0, 100 - min(per_hr * 8, 60) - avg * 0.4) if trips else None
        reasons = {}
        for t in trips:
            for k, v in t["alarms"].items():
                reasons[k] = reasons.get(k, 0) + v
        drivers[name] = {
            "driver": name, "trips": len(trips), "hours": round(hours, 2),
            "today_hours": duty_hours_today(name), "alarms": alarms, "warnings": warns,
            "alarms_per_hour": round(per_hr, 2), "avg_fatigue": round(avg, 1),
            "avg_kss_end": round(sum(kss) / len(kss), 1) if kss else None,
            "false_alarms": sum(t.get("false_alarms", 0) for t in trips),
            "emergencies": sum(t.get("emergency_alerts", 0) for t in trips),
            "last_trip": max((t["end"] for t in trips), default=None),
            "safety_score": None if safety is None else round(safety, 1),
            "alarm_reasons": reasons,
        }
    rows = list(drivers.values())
    return {
        "duty_limit_hours": SETTINGS["duty_limit_hours"], "current_driver": SETTINGS["driver_name"],
        "drivers": rows,
        "totals": {"drivers": len([r for r in rows if r["trips"]]), "trips": len(TRIPS),
                   "hours": round(sum(r["hours"] for r in rows), 2), "alarms": sum(r["alarms"] for r in rows),
                   "over_limit_today": [r["driver"] for r in rows if r["today_hours"] >= SETTINGS["duty_limit_hours"]]},
    }


class FleetQuestion(BaseModel):
    question: str


@app.get("/api/fleet")
def get_fleet():
    return fleet_summary()


@app.post("/api/fleet/ask")
async def fleet_ask(q: FleetQuestion):
    """Ask the fleet data a question in plain English (Gemini turns it into an answer)."""
    question = q.question.strip()[:500]
    if not question:
        raise HTTPException(400, "empty question")
    key = SETTINGS["gemini_api_key"].strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {"source": "none", "answer": "Add a Gemini API key in Settings to ask questions in plain English. "
                                           "The fleet table above already shows the main numbers."}
    data = fleet_summary()
    data["recent_trips"] = [{k: v for k, v in t.items() if k not in ("timeline", "start_location", "end_location")}
                            for t in TRIPS[-50:]]
    prompt = ("You are a fleet-safety analyst. Answer the manager's question using ONLY this JSON data from a "
              "driver drowsiness monitoring system. Be concise (max 5 sentences), cite numbers, and say so if the "
              "data cannot answer it. Times are Unix seconds.\n\nQuestion: " + question + "\n\nData:\n"
              + json.dumps(data)[:60000])
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{SETTINGS['gemini_model']}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        resp = await asyncio.to_thread(lambda: json.loads(urllib.request.urlopen(req, timeout=30).read()))
        return {"source": "ai", "answer": resp["candidates"][0]["content"]["parts"][0]["text"].strip()}
    except Exception as exc:
        return {"source": "error", "answer": f"AI unavailable: {exc}"}


# ------------------------------------------------------- rest-stop proxy ----
OVERPASS_SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter",
                    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
_OVERPASS_CACHE: dict[str, tuple[float, dict]] = {}


def _overpass_fetch(url: str, q: str, timeout: float) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "DriveAlert/1.0 (student project)"})
    data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if not isinstance(data.get("elements"), list):
        raise ValueError("bad answer")
    return data


@app.get("/api/overpass")
async def overpass_proxy(q: str):
    """Rest-stop search through the server: asks all Overpass mirrors at once, returns the first
    answer and caches it for 6 hours (works even if the browser cannot reach a mirror)."""
    if not q.startswith("[out:json]") or len(q) > 2000:
        raise HTTPException(400, "only Overpass JSON queries")
    hit = _OVERPASS_CACHE.get(q)
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    tasks = [asyncio.create_task(asyncio.to_thread(_overpass_fetch, u, q, 14)) for u in OVERPASS_SERVERS]
    try:
        for fut in asyncio.as_completed(tasks, timeout=15):
            try:
                data = await fut
            except Exception:
                continue
            if len(_OVERPASS_CACHE) > 300:
                _OVERPASS_CACHE.clear()
            _OVERPASS_CACHE[q] = (time.time(), data)
            return data
    except asyncio.TimeoutError:
        pass
    raise HTTPException(504, "OpenStreetMap servers did not answer")


# ------------------------------------------------------------ SUS survey ----
SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


class SusResponse(BaseModel):
    answers: list[int]
    name: str | None = None
    role: str | None = None
    comment: str | None = None


def sus_score(answers):
    """Standard SUS: odd items (positive) score a-1, even items (negative) 5-a; sum x 2.5 -> 0-100."""
    return sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers)) * 2.5


def sus_grade(score):
    if score >= 85:
        return "Excellent"
    if score >= 72:
        return "Good"
    if score >= 52:
        return "OK"
    return "Poor"


@app.get("/api/sus")
def get_sus():
    rows = _load(SUS_FILE, [])
    scores = [r["score"] for r in rows]
    mean = float(np.mean(scores)) if scores else None
    rows = [{**r, "grade": sus_grade(r["score"])} for r in rows]
    return {"questions": SUS_QUESTIONS, "responses": rows[::-1],
            "summary": {"n": len(scores), "mean": None if mean is None else round(mean, 1),
                        "sd": round(float(np.std(scores, ddof=1)), 1) if len(scores) > 1 else None,
                        "grade": None if mean is None else sus_grade(mean),
                        "above_average": None if mean is None else mean > 68,
                        "item_means": [round(float(np.mean([r["answers"][i] for r in rows])), 2) for i in range(10)]
                        if rows else []}}


@app.post("/api/sus")
def post_sus(r: SusResponse):
    if len(r.answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in r.answers):
        raise HTTPException(400, "answer all 10 questions with 1-5")
    rows = _load(SUS_FILE, [])
    score = sus_score(r.answers)
    rows.append({"time": time.time(), "name": (r.name or "").strip()[:60], "role": (r.role or "").strip()[:60],
                 "comment": (r.comment or "").strip()[:500], "answers": r.answers, "score": score})
    _save(SUS_FILE, rows)
    return {"score": score, "grade": sus_grade(score)}


@app.delete("/api/sus")
def clear_sus():
    _save(SUS_FILE, [])
    return {"cleared": True}


@app.get("/api/model-info")
def model_info():
    """Training results written by train_model.py (for the About page)."""
    info = {}
    for task, model_path in (("eye", EYE_MODEL), ("yawn", YAWN_MODEL)):
        f = MODEL_DIR / f"{task}_metrics.json"
        if not any((MODEL_DIR / f"{task}_{k}.json").exists() for k in ("metrics", "classical_metrics", "dl_metrics")):
            continue
        m = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        entry = {k: m.get(k) for k in ("classes", "img_size", "test_metrics", "counts", "split_sizes",
                                       "confusion_matrix", "classification_report", "history",
                                       "fine_tune_start")}
        entry["plots"] = [p for p in (f"{task}_confusion_matrix.png", f"{task}_training_curves.png")
                          if (MODEL_DIR / p).exists()]
        if Path(model_path).exists():
            entry["size_mb"] = round(Path(model_path).stat().st_size / 1e6, 2)
            model = load_keras_model(model_path)
            if model is not None:
                entry["params"] = int(model.count_params())
        cl = MODEL_DIR / f"{task}_classical_metrics.json"
        if cl.exists():
            c = json.loads(cl.read_text(encoding="utf-8"))
            entry["classical"] = {k: c.get(k) for k in ("best_classical", "features", "train_size",
                                                       "test_size", "results", "confusion_matrices")}
            entry["classical"]["plots"] = [p for p in (f"{task}_model_comparison.png", f"{task}_rf_importance.png")
                                           if (MODEL_DIR / p).exists()]
        dl = MODEL_DIR / f"{task}_dl_metrics.json"
        if dl.exists():
            entry["dl"] = json.loads(dl.read_text(encoding="utf-8"))
            if (MODEL_DIR / f"{task}_dl_comparison.png").exists():
                entry["dl"]["plot"] = f"{task}_dl_comparison.png"
        for key, fname in (("significance", f"{task}_significance.json"), ("failures", f"{task}_failures.json")):
            if (MODEL_DIR / fname).exists():
                entry[key] = json.loads((MODEL_DIR / fname).read_text(encoding="utf-8"))
        if (MODEL_DIR / f"{task}_failures.png").exists() and "failures" in entry:
            entry["failures"]["plot"] = f"{task}_failures.png"
        tfl = MODEL_DIR / f"{task}_model_int8.tflite"
        if tfl.exists():
            entry["tflite_mb"] = round(tfl.stat().st_size / 1e6, 2)
        info[task] = entry
    ft = MODEL_DIR / "fusion_tuning.json"
    if ft.exists():
        info["fusion_tuning"] = json.loads(ft.read_text(encoding="utf-8"))
        if (MODEL_DIR / "fusion_tuning.png").exists():
            info["fusion_tuning"]["plot"] = "fusion_tuning.png"
    nm = MODEL_DIR / "eye_night_compare.json"
    if nm.exists():
        info["night"] = json.loads(nm.read_text(encoding="utf-8"))
    tm = MODEL_DIR / "temporal_metrics.json"
    if tm.exists():
        info["temporal"] = json.loads(tm.read_text(encoding="utf-8"))
    return info


@app.get("/api/model-plot/{name}")
def model_plot(name: str):
    path = MODEL_DIR / name
    if name not in PLOTS or not path.exists():
        raise HTTPException(404, "plot not found")
    return FileResponse(path)


# ------------------------------------------------------------ WebSocket ----
def decode_frame(data: str):
    """Decode a base64 JPEG/PNG (optionally a data URL). Returns None if invalid."""
    try:
        if "," in data[:100]:
            data = data.split(",", 1)[1]  # strip "data:image/jpeg;base64,"
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (ValueError, TypeError):  # binascii.Error is a ValueError
        return None


async def _emergency(ws: WebSocket, text: str):
    ok, msg = await asyncio.to_thread(send_telegram, text)
    try:
        await ws.send_json({"type": "emergency", "ok": ok, "message": msg})
    except Exception:
        pass


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    det = DrowsinessDetector(DetectorConfig(**asdict(CONFIG)), EYE_MODEL, YAWN_MODEL)
    profile = SETTINGS["profiles"].get(SETTINGS["driver_name"])
    if SETTINGS["adaptive_learning"] and profile:
        det.load_profile(profile)
    log = SessionLog()
    SESSIONS.add(det)
    await ws.send_json({"type": "info", "message": "connected", "session": log.session,
                        "profile": profile if SETTINGS["adaptive_learning"] else None,
                        "duty_hours_today": duty_hours_today(SETTINGS["driver_name"]),
                        "duty_limit_hours": SETTINGS["duty_limit_hours"]})
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue
            kind = msg.get("type", "frame")
            if kind == "recalibrate":
                det.recalibrate()
                await ws.send_json({"type": "info", "message": "recalibrating"})
                continue
            if kind == "ack":
                await ws.send_json({"type": "info", "message": "acknowledged" if log.ack()
                                    else "no active alarm"})
                continue
            if kind == "location":
                try:
                    log.set_location(msg["lat"], msg["lon"], msg.get("accuracy"))
                    if msg.get("speed") is not None:   # km/h from the browser GPS
                        det.speed_kmh = max(0.0, float(msg["speed"]))
                except (KeyError, TypeError, ValueError):
                    await ws.send_json({"type": "error", "message": "bad location"})
                continue
            if kind == "speed":   # manual / simulated speed for testing the gate
                try:
                    det.speed_kmh = None if msg.get("value") is None else max(0.0, float(msg["value"]))
                except (TypeError, ValueError):
                    pass
                continue
            if kind == "false_alarm":
                reason = log.mark_false_alarm()
                if reason and SETTINGS["adaptive_learning"]:
                    prof = det.apply_feedback(reason)
                    prof["false_alarms"] = SETTINGS["profiles"].get(SETTINGS["driver_name"], {}).get("false_alarms", 0) + 1
                    SETTINGS["profiles"][SETTINGS["driver_name"]] = prof
                    save_settings()
                    await ws.send_json({"type": "info", "message": "false_alarm_learned",
                                        "reason": reason, "profile": prof})
                else:
                    await ws.send_json({"type": "info", "message": "false_alarm_logged" if reason
                                        else "no recent alarm to mark"})
                continue
            if kind == "crash":
                g = float(msg.get("g", 0) or 0)
                log.crash(g)
                if SETTINGS["emergency_enabled"]:
                    log.emergencies += 1
                    asyncio.create_task(_emergency(ws, log.crash_text(g)))
                else:
                    await ws.send_json({"type": "emergency", "ok": False,
                                        "message": "emergency alerts are disabled in Settings"})
                continue
            if kind == "night_event":
                reason = msg.get("reason")
                if reason not in NIGHT_EVENTS:
                    await ws.send_json({"type": "error", "message": "unknown night event"})
                    continue
                log.night_event(reason, msg.get("detail"))
                if reason == "checkin_missed" and log.missed_checkins >= 2 and SETTINGS["emergency_enabled"] \
                        and time.time() - log.last_emergency > SETTINGS["emergency_window_min"] * 60:
                    log.last_emergency = time.time()
                    log.emergencies += 1
                    asyncio.create_task(_emergency(ws, log.checkin_text()))
                continue
            if kind == "kss":
                try:
                    log.set_kss(msg["value"], msg.get("phase", "start"))
                    await ws.send_json({"type": "info", "message": "kss saved"})
                except (KeyError, TypeError, ValueError):
                    await ws.send_json({"type": "error", "message": "KSS must be 1-9"})
                continue
            if kind != "frame" or "image" not in msg:
                await ws.send_json({"type": "error", "message": "unknown message"})
                continue

            frame = decode_frame(msg["image"])
            if frame is None:
                await ws.send_json({"type": "error", "message": "bad image"})
                continue
            t0 = time.perf_counter()
            result = await asyncio.to_thread(det.process, frame)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            result["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            result["session"] = log.session
            result["type"] = "result"
            log.update(result)
            if log.emergency_due():
                result["emergency"] = "sending"
                asyncio.create_task(_emergency(ws, log.emergency_text(result)))
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        log.close()
        SESSIONS.discard(det)
        det.close()


FRONTEND = BASE / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
