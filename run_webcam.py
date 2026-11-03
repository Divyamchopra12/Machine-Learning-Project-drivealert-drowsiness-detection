"""
Real-time drowsiness detection from a webcam (desktop app, no frontend needed).

  python run_webcam.py                # default camera, CNNs if models/ exist
  python run_webcam.py --camera 1 --no-cnn

Keys:  q = quit   r = recalibrate (look ahead with eyes open for 3 s)

The alarm escalates with the detector's alarm_level:
  1 early warning  -> soft low chime every 3 s
  2 alarm          -> 880 Hz beep every 0.6 s
  3 critical alarm -> fast high-pitched 1300 Hz beeps
"""
import argparse
import threading
import time

import cv2
import numpy as np

from detector import DetectorConfig, DrowsinessDetector

# level: (frequency Hz, tone length s, pause between tones s, volume 0-1)
LEVELS = {1: (520, 0.25, 3.0, 0.25), 2: (880, 0.35, 0.6, 0.6), 3: (1300, 0.18, 0.12, 0.9)}


class Alarm:
    """Plays an escalating alarm in a background thread (pygame, then winsound, else silent)."""

    def __init__(self):
        self.level = 0
        self._play = self._init_backend()
        threading.Thread(target=self._loop, daemon=True).start()

    def _init_backend(self):
        try:
            import pygame

            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            sounds = {}
            for lvl, (freq, dur, _, vol) in LEVELS.items():
                t = np.linspace(0, dur, int(44100 * dur), False)
                tone = np.sin(2 * np.pi * freq * t) * np.minimum(1, (dur - t) * 40)  # soft release
                sounds[lvl] = pygame.sndarray.make_sound((tone * vol * 32767).astype(np.int16))
            return lambda lvl: sounds[lvl].play()
        except Exception:
            pass
        try:
            import winsound

            return lambda lvl: winsound.Beep(LEVELS[lvl][0], int(LEVELS[lvl][1] * 1000))
        except Exception:
            print("[alarm] no audio backend found; visual alarm only")
            return None

    def _loop(self):
        while True:
            lvl = self.level
            if lvl and self._play:
                self._play(lvl)
                time.sleep(LEVELS[lvl][1] + LEVELS[lvl][2])
            else:
                time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-cnn", action="store_true", help="landmarks only (no TensorFlow)")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    cfg = DetectorConfig(use_cnn=not args.no_cnn, calibrate=not args.no_calibration)
    det = DrowsinessDetector(cfg)
    print("CNN models loaded:", det.cnn_loaded)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    alarm = Alarm()
    prev, fps = time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

            result = det.process(frame)
            alarm.level = result["alarm_level"]
            det.annotate(frame, result)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"{fps:.0f} FPS  L{result['alarm_level']}", (frame.shape[1] - 140, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if result["calibrating"]:
                cv2.putText(frame, "Calibrating: look ahead, eyes open",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 180, 0), 2)

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                det.recalibrate()
    finally:
        alarm.level = 0
        cap.release()
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
