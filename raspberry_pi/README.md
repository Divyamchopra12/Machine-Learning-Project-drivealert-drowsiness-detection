# DriveAlert – Raspberry Pi in-car unit

A stand-alone version that runs in the car with no laptop and no internet. It uses the same
`detector.py` as the web app and drives a buzzer, a vibration motor and two LEDs according to
the alarm level (L0–L3).

## Parts (approximate prices in India, check before buying)

| Part | Notes |
|---|---|
| Raspberry Pi 4 (4 GB) or Pi 5 | + 5 V 3 A USB-C supply (or a car USB-C adapter) |
| USB webcam **or** Pi Camera Module 3 NoIR | NoIR + 850 nm IR LED ring = works at night (train_night.py gives the matching IR eye model; use --cnn) |
| Active buzzer module (5 V) | loud enough for a cabin |
| Coin vibration motor + 2N2222 / BC547 transistor + 1 kΩ resistor + 1N4007 diode | for seat / steering vibration |
| Red + green LED with 220 Ω resistors | status |
| MQ-3 alcohol sensor module (optional) | digital output (DO) |
| Breadboard + jumper wires | |

## Wiring (BCM pin numbers)

```
Buzzer  +  → GPIO 17 (pin 11)      Buzzer − → GND (pin 9)
Motor:  GPIO 27 (pin 13) → 1 kΩ → transistor base
        transistor collector → motor − ; motor + → 5 V (pin 2) ; emitter → GND
        diode across the motor (stripe to 5 V) protects the Pi
Green LED → 220 Ω → GPIO 22 (pin 15) ;  Red LED → 220 Ω → GPIO 23 (pin 16) ; LED − → GND
MQ-3:  VCC → 5 V (pin 4), GND → GND (pin 6), DO → GPIO 24 (pin 18)
       (the module's DO is open-collector/LOW-active, so it is safe for the 3.3 V pin with the
        internal pull-up; never connect the analog AO pin directly to the Pi)
```

## Install on Raspberry Pi OS (64-bit, Bookworm)

```bash
sudo apt update && sudo apt install -y python3-venv python3-picamera2 libgl1
cd ~/drowsiness-detection
python3 -m venv --system-site-packages venv      # system packages give picamera2 + gpiozero
source venv/bin/activate
pip install "numpy<2" opencv-python mediapipe scikit-learn
# TensorFlow is only needed with --cnn:  pip install tensorflow
```

MediaPipe publishes wheels for 64-bit Raspberry Pi OS (aarch64). If `pip install mediapipe` fails,
check that you are on the 64-bit OS and a supported Python version.

## Run

```bash
python raspberry_pi/run_pi.py                  # USB camera, EAR + head pose (fastest)
python raspberry_pi/run_pi.py --picam --show   # Pi camera with preview window
python raspberry_pi/run_pi.py --cnn            # add the CNN models (slower on a Pi 4)
python raspberry_pi/run_pi.py --mq3            # alcohol check (20 s warm-up) before starting
python raspberry_pi/run_pi.py --camera clip.mp4 --frames 300   # test with a video file
```

On a normal PC there is no GPIO, so it runs in **console mode** and prints what the buzzer,
motor and LEDs would do – useful for testing before you build the hardware.

## Start automatically when the car starts

```bash
crontab -e
# add:
@reboot cd /home/pi/drowsiness-detection && venv/bin/python raspberry_pi/run_pi.py --picam >> pi.log 2>&1
```

## Honest limitations

- Measure your own FPS on the Pi and report that number; do not quote a figure you have not measured.
- The MQ-3 is a hobby sensor: it shows that alcohol is present, it does **not** measure blood
  alcohol and is not a legal breathalyser.
- For the full web dashboard on the Pi, run `python -m uvicorn app:app --host 0.0.0.0 --port 8000`
  and open it from a phone on the same Wi-Fi.
