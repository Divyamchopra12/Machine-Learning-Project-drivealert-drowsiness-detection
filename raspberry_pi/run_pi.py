"""
DriveAlert on a Raspberry Pi – a stand-alone in-car unit, no laptop or internet needed.

Hardware (see raspberry_pi/README.md for wiring):
  camera          USB webcam, or Pi Camera Module (use --picam; NoIR camera + IR LEDs for night)
  buzzer          active buzzer on GPIO 17
  vibration motor coin motor via NPN transistor on GPIO 27 (seat / steering wheel)
  LEDs            green GPIO 22 (OK), red GPIO 23 (alarm)
  MQ-3 (optional) alcohol sensor digital output on GPIO 24 – checked before the trip starts

Escalating outputs follow the same alarm levels as the web app:
  L0 normal      green LED
  L1 early warn  short beep every 3 s, red LED blinks slowly
  L2 alarm       fast beeps + vibration pulses, red LED on
  L3 critical    continuous buzzer + continuous vibration, red LED flashes

On a PC (no GPIO) it runs in "console mode" and prints the outputs instead, so you can test it anywhere.

Run (from the project folder):
  python raspberry_pi/run_pi.py                 # USB camera, landmarks only (fastest on a Pi)
  python raspberry_pi/run_pi.py --picam --show  # Pi camera + preview window
  python raspberry_pi/run_pi.py --cnn           # also use the CNN models (slower on a Pi 4)
  python raspberry_pi/run_pi.py --mq3           # alcohol check before starting
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from detector import DetectorConfig, DrowsinessDetector  # noqa: E402


# --------------------------------------------------------------------- outputs
class Outputs:
    """Buzzer, vibration motor and LEDs, driven by the alarm level in a background thread."""

    def __init__(self, buzzer=17, motor=27, green=22, red=23):
        self.level, self.running, self.console = 0, True, False
        try:
            from gpiozero import LED, Buzzer, OutputDevice

            self.buzzer, self.motor = Buzzer(buzzer), OutputDevice(motor)
            self.green, self.red = LED(green), LED(red)
            print(f"[gpio] buzzer={buzzer} motor={motor} green={green} red={red}")
        except Exception as exc:  # not a Pi, or gpiozero missing
            self.console = True
            print(f"[gpio] not available ({exc.__class__.__name__}) – console mode, outputs are printed")
        threading.Thread(target=self._loop, daemon=True).start()

    def _set(self, buzz=False, vib=False, red=False, green=False):
        if self.console:
            return
        for dev, on in ((self.buzzer, buzz), (self.motor, vib), (self.red, red), (self.green, green)):
            dev.on() if on else dev.off()

    def _loop(self):
        last_printed = None
        while self.running:
            lvl = self.level
            if self.console and lvl != last_printed:
                print(["[out] L0 normal – green LED", "[out] L1 early warning – soft beep every 3 s",
                       "[out] L2 ALARM – fast beeps + vibration", "[out] L3 CRITICAL – siren + continuous vibration"][lvl])
                last_printed = lvl
            if lvl == 0:
                self._set(green=True)
                time.sleep(0.2)
            elif lvl == 1:
                self._set(buzz=True, red=True)
                time.sleep(0.15)
                self._set(red=False)
                time.sleep(2.85)
            elif lvl == 2:
                self._set(buzz=True, vib=True, red=True)
                time.sleep(0.3)
                self._set(red=True)
                time.sleep(0.3)
            else:
                self._set(buzz=True, vib=True, red=True)
                time.sleep(0.15)
                self._set(buzz=True, vib=True)
                time.sleep(0.15)

    def close(self):
        self.running = False
        time.sleep(0.3)
        self._set()


# --------------------------------------------------------------------- MQ-3
def alcohol_check(pin=24, warmup_s=20, sample_s=10) -> bool:
    """True if alcohol was detected. Most MQ-3 modules pull DO LOW above the threshold set by their
    potentiometer. The sensor needs to warm up; the driver breathes on it during sampling."""
    try:
        from gpiozero import DigitalInputDevice
    except Exception:
        print("[mq3] gpiozero not available – skipping alcohol check")
        return False
    sensor = DigitalInputDevice(pin, pull_up=True)
    print(f"[mq3] warming up {warmup_s} s …")
    time.sleep(warmup_s)
    print(f"[mq3] breathe towards the sensor for {sample_s} s")
    end, hits, n = time.time() + sample_s, 0, 0
    while time.time() < end:
        hits += sensor.value == 0
        n += 1
        time.sleep(0.1)
    sensor.close()
    detected = hits / max(n, 1) > 0.3
    print("[mq3] ALCOHOL DETECTED – do not drive" if detected else "[mq3] OK")
    return detected


# --------------------------------------------------------------------- camera
class Camera:
    def __init__(self, picam=False, index=0, width=640, height=480):
        self.picam = None
        if picam:
            from picamera2 import Picamera2

            self.picam = Picamera2()
            self.picam.configure(self.picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}))
            self.picam.start()
        else:
            self.cap = cv2.VideoCapture(int(index) if str(index).isdigit() else index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open camera {index}")

    def read(self):
        if self.picam:
            return True, self.picam.capture_array()   # RGB888 in picamera2 is already BGR order for OpenCV
        return self.cap.read()

    def close(self):
        if self.picam:
            self.picam.stop()
        else:
            self.cap.release()


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="DriveAlert Raspberry Pi unit")
    ap.add_argument("--picam", action="store_true", help="use the Pi Camera Module (picamera2)")
    ap.add_argument("--camera", default="0", help="USB camera index, or a video file to test with")
    ap.add_argument("--cnn", action="store_true", help="also run the CNN models (slower)")
    ap.add_argument("--show", action="store_true", help="show a preview window (needs a display)")
    ap.add_argument("--mq3", action="store_true", help="alcohol check before starting")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument("--buzzer", type=int, default=17)
    ap.add_argument("--motor", type=int, default=27)
    args = ap.parse_args()

    out = Outputs(args.buzzer, args.motor)
    if args.mq3 and alcohol_check():
        out.level = 3
        time.sleep(5)
        out.close()
        raise SystemExit(1)

    cfg = DetectorConfig(use_cnn=args.cnn, eye_mode="fusion" if args.cnn else "ear",
                         use_temporal=False, phone_detection=False)
    det = DrowsinessDetector(cfg, str(ROOT / "models/eye_model.keras"), str(ROOT / "models/yawn_model.keras"))
    cam = Camera(args.picam, args.camera)
    print("Running – Ctrl+C to stop. Look straight ahead for 3 s to calibrate.")
    n, t0, last_status = 0, time.time(), None
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            r = det.process(frame)
            out.level = r["alarm_level"]
            n += 1
            if r["status"] != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] {r['status']:<11} score {r['fatigue_score']:5.1f} "
                      f"L{r['alarm_level']} {r.get('alarm_reason') or ''}")
                last_status = r["status"]
            if args.show:
                det.annotate(frame, r)
                cv2.imshow("DriveAlert Pi", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"{n} frames, {n / max(time.time() - t0, 1e-6):.1f} FPS")
        out.close()
        cam.close()
        det.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
