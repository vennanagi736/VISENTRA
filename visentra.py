import argparse
import csv
import faulthandler
import json
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import socket
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# VISENTRA PATHS
# ============================================================
ROOT = Path(r"V:\visentra")
ADB_EXE = ROOT / r"platform-tools-latest-windows\platform-tools\adb.exe"
SCRCPY_EXE = Path(
    r"V:\visentra\scrcpy\scrcpy.exe"
)
RF_PYTHON = ROOT / r"environments\rfdetr_env\Scripts\python.exe"
LANE_PYTHON = ROOT / r"environments\condlanenet_env\Scripts\python.exe"
RF_MODEL = ROOT / r"models\rf-detr-medium.pth"
LANE_MODEL = ROOT / r"models\curvelanes_medium.pth"
COND_LANE_ROOT = ROOT / "CondLaneNet"
LANE_CONFIG = COND_LANE_ROOT / r"configs\condlanenet\curvelanes\curvelanes_medium_test.py"

# Metric3D runs in the existing MonoPilot environment, not the RF-DETR env.
METRIC_PYTHON = ROOT / r"MonoPilot\.venv\Scripts\python.exe"
METRIC_MODEL = ROOT / r"metric3d_test\models\metric3d_v2_vit_small.onnx"
METRIC_CONFIG = ROOT / r"metric3d_test\models\preprocessor_config.json"
METRIC_RESULTS_DIR = ROOT / r"metric3d_test\results"

OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
# Per-session diagnostic logs: one file per worker plus one for the parent process.
WORKER_LOG_DIR = OUTPUT_DIR / "worker_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
METRIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)

SESSION_TAG = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
PARENT_LOG_PATH = WORKER_LOG_DIR / f"{SESSION_TAG}_parent.log"
PERF_CSV_PATH = OUTPUT_DIR / f"perf_{SESSION_TAG}.csv"

# Provisional empirically-fitted factor from the initial 5/10/15/20 m samples.
# It is NOT an independently validated camera calibration.
DEFAULT_DEPTH_SCALE = 0.373572
DEPTH_CAP_VALUE = 199.5

# ============================================================
# ADDITIONAL MODELS (both OFF by default: they cost CPU and are not part of the core pipeline)
#   MobileNetV3 : enable with --with-mobilenet
#   DeepSORT    : enable with --with-deepsort (ByteTrack always runs)
# ============================================================
# The flag travels to worker processes through an environment variable.
MOBILENETV3_ENABLED = os.environ.get("VISENTRA_MOBILENET", "0") == "1"
MOBILENETV3_EVERY_N_FRAMES = 10  # Reduce CPU load during live inference
MOBILENETV3_MAX_CROPS = 8

def draw_reverse_grid(frame):
    h, w = frame.shape[:2]

    # Fixed display area for the reversing-camera grid
    top_y = int(h * 0.28)
    bottom_y = int(h * 0.94)
    top_left = int(w * 0.36)
    top_right = int(w * 0.64)
    bottom_left = int(w * 0.08)
    bottom_right = int(w * 0.92)
    
    # Perspective boundary
    cv2.line(frame, (top_left, top_y),
             (bottom_left, bottom_y), (0, 255, 255), 2)
    cv2.line(frame, (top_right, top_y),
             (bottom_right, bottom_y), (0, 255, 255), 2)

    # Left / center / right dividers
    for fraction in (1 / 3, 2 / 3):
        top_x = int(top_left + fraction * (top_right - top_left))
        bottom_x = int(
            bottom_left + fraction * (bottom_right - bottom_left)
        )
        cv2.line(frame, (top_x, top_y),
                 (bottom_x, bottom_y), (0, 200, 255), 1)

    # Ten visual bands across the nominal 0–100 m range
    for i in range(1, 10):
        fraction = i / 10
        y = int(bottom_y - fraction * (bottom_y - top_y))
        left_x = int(bottom_left + fraction * (top_left - bottom_left))
        right_x = int(bottom_right + fraction * (top_right - bottom_right))

        cv2.line(frame, (left_x, y), (right_x, y),
                 (0, 180, 180), 1)

        distance_label = f"{i * 10} m"
        cv2.putText(
            frame, distance_label,
            (left_x + 4, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    cv2.putText(
        frame, "LEFT", (int(w * 0.17), int(h * 0.90)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.putText(
        frame, "CENTER", (int(w * 0.43), int(h * 0.90)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.putText(
        frame, "RIGHT", (int(w * 0.69), int(h * 0.90)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    return frame
def set_mobilenet_enabled(flag):
    global MOBILENETV3_ENABLED
    MOBILENETV3_ENABLED = bool(flag)
    os.environ["VISENTRA_MOBILENET"] = "1" if flag else "0"


# Starting point for CPU thread allocation on the i5-1335U (2 P-cores + 8 E-cores, 12 logical).
# These numbers are NOT benchmarked: change them with --threads rf,lane,metric and compare the
# measured FPS / latency printed by this script.
DEFAULT_THREADS = {"rf": 4, "lane": 2, "metric": 3}

_MOBILENETV3_MODEL = None
_MOBILENETV3_TRANSFORM = None
_MOBILENETV3_LABELS = None

# Only estimate/display distance for road-vehicle classes. Other RF-DETR
# detections may still be drawn, but they are not treated as vehicle targets.
VEHICLE_CLASS_NAMES = {
    "car", "truck", "bus", "motorcycle", "motorbike",
    "bicycle", "bike", "van", "vehicle", "train", "tram",
}


def is_vehicle_class(class_name):
    """Return True only for labels considered vehicles by this demo."""
    normalized = str(class_name or "").strip().lower().replace("_", " ")
    return normalized in VEHICLE_CLASS_NAMES


_RF_DETECTOR = None
_LANE_DETECTOR = None
_LANE_CFG = None
_LANE_PIPELINE = None
_LANE_POSTPROCESSOR = None
_METRIC_ESTIMATOR = None

# ============================================================
# LOGGING HELPERS
# ============================================================
_PARENT_LOG_ENABLED = False
_PARENT_LOG_LOCK = threading.Lock()


def _timestamp():
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


def parent_log(message):
    """Append a line to the parent's per-session log file (live mode only)."""
    if not _PARENT_LOG_ENABLED:
        return
    line = f"{_timestamp()} [parent pid {os.getpid()}] {message}\n"
    try:
        with _PARENT_LOG_LOCK, open(PARENT_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


def tail_text(path, lines=25):
    """Last lines of a text file, or '' if it cannot be read."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines)).rstrip()
    except OSError:
        return ""


def log_info(message):
    print(f"[VISENTRA] {message}", flush=True)
    parent_log(f"INFO {message}")


def log_warn(message):
    """Informational warning: the program keeps running."""
    print(f"[VISENTRA][WARNING] {message}", file=sys.stderr, flush=True)
    parent_log(f"WARNING {message}")


def log_error(message):
    """Fatal error: the program is about to stop."""
    print(f"[VISENTRA][ERROR] {message}", file=sys.stderr, flush=True)
    parent_log(f"ERROR {message}")


_LOGGED_ONCE = set()


def log_warn_once(key, message):
    if key in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(key)
    log_warn(message)


# ============================================================
# SHUTDOWN / FIRST-FAILURE TRACKING
# ============================================================
class ShutdownTracker:
    """Remembers WHY the live session ended and WHAT failed first.

    On Windows, Popen.terminate() ends a child with exit code 1, so "exit code 1" from a
    worker proves nothing by itself. This tracker records, with timestamps, whether a worker
    exited on its own before VISENTRA started shutting down (a real failure) or only after
    (a consequence of cleanup). Only the first reason / first failure is kept.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.event = threading.Event()
            self.t0 = time.monotonic()
            self.reason = None
            self.began_at = None
            self.first_failure = None
            self.first_failure_at = None

    def is_set(self):
        return self.event.is_set()

    def begin(self, reason):
        """Record the original shutdown reason; later calls do not overwrite it."""
        with self._lock:
            if self.event.is_set():
                return False
            self.reason = reason
            self.began_at = time.monotonic()
            self.event.set()
        parent_log(f"SHUTDOWN BEGINS at +{self.began_at - self.t0:.2f}s: {reason}")
        return True

    def failure(self, detail):
        """Record a failure that happened while the session was still running."""
        with self._lock:
            first = self.first_failure is None
            if first:
                self.first_failure = detail
                self.first_failure_at = time.monotonic()
            elapsed = time.monotonic() - self.t0
        if first:
            log_error(f"FIRST FAILURE (at +{elapsed:.2f}s, before shutdown began): {detail}")
        else:
            parent_log(f"secondary failure at +{elapsed:.2f}s: {detail}")
        return first


SHUTDOWN = ShutdownTracker()


class WorkerExitedError(RuntimeError):
    """A worker process ended on its own while the session was running."""


class WorkerShutdownError(RuntimeError):
    """A worker ended because VISENTRA was already shutting down (not a failure)."""


# ============================================================
# MEASURED PERFORMANCE STATISTICS (rolling windows + whole-session totals)
# ============================================================
class RateMeter:
    """Events per second over the last `window` events."""

    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self.total = 0

    def tick(self, now=None):
        self._times.append(time.perf_counter() if now is None else now)
        self.total += 1

    def fps(self):
        if len(self._times) < 2:
            return None
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else None


class RollingStat:
    """Rolling mean over the last `window` samples plus a whole-session mean."""

    def __init__(self, window=30):
        self._values = deque(maxlen=window)
        self._sum = 0.0
        self._n = 0

    def add(self, value):
        self._values.append(float(value))
        self._sum += float(value)
        self._n += 1

    def mean(self):
        return sum(self._values) / len(self._values) if self._values else None

    def mean_all(self):
        return self._sum / self._n if self._n else None


class PerfStats:
    """Thread-safe collection of the measured rates, latencies and counters."""

    RATES = ("capture", "display", "rf", "lane", "metric")
    LATENCIES = ("rf_worker", "rf_roundtrip", "lane_worker", "lane_roundtrip",
                 "metric_worker", "metric_roundtrip", "bytetrack", "deepsort", "e2e")
    COUNTERS = ("captured", "dropped_rf", "dropped_lane", "dropped_metric", "dropped_track",
                "video_skipped")

    def __init__(self):
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._rates = {name: RateMeter() for name in self.RATES}
        self._lat = {name: RollingStat() for name in self.LATENCIES}
        self._counts = {name: 0 for name in self.COUNTERS}

    def tick(self, name, now=None):
        with self._lock:
            self._rates[name].tick(now)

    def add(self, name, value):
        with self._lock:
            self._lat[name].add(value)

    def count(self, name, amount=1):
        with self._lock:
            self._counts[name] += amount

    def fps(self, name):
        with self._lock:
            return self._rates[name].fps()

    def ms(self, name):
        with self._lock:
            return self._lat[name].mean()

    def ms_all(self, name):
        with self._lock:
            return self._lat[name].mean_all()

    def counter(self, name):
        with self._lock:
            return self._counts[name]

    def total(self, name):
        with self._lock:
            return self._rates[name].total


def _fmt(value, spec, unit=""):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec) + unit


class ResourceMonitor:
    """Whole-machine CPU % and RAM of this process tree. Needs psutil (optional).

    The parent is deliberately not raised to ABOVE_NORMAL priority: on this CPU-only
    pipeline that can compete with the model worker processes for processor time.
    """

    def __init__(self):
        self.available = False
        self.cpu = None
        self.ram_mb = None
        self._last = 0.0
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
            psutil.cpu_percent(None)  # prime the counter
            self.available = True
        except ImportError:
            pass

    def sample(self, now, every=2.0):
        if not self.available or now - self._last < every:
            return
        self._last = now
        try:
            self.cpu = self._psutil.cpu_percent(None)
            total = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except self._psutil.Error:
                    pass
            self.ram_mb = total / 1048576.0
        except Exception:  # noqa: BLE001 - monitoring must never break the pipeline
            pass


# ============================================================
# AUTOMATIC VIVO V27 REAR CAMERA STARTUP (scrcpy raw H.264)
# ============================================================
SCRCPY_SERVER_VERSION_FALLBACK = "5.0.1"
# Session-specific file name so we never overwrite a jar used by a separate scrcpy run.
VIVO_REMOTE_SERVER_PATH = "/data/local/tmp/visentra-scrcpy-server.jar"
VIVO_CONNECT_TIMEOUT_S = 25.0       # server start + socket + first encoded bytes
VIVO_FIRST_FRAME_TIMEOUT_S = 20.0   # first decoded frame after the first bytes
VIVO_READ_TIMEOUT_S = 8.0           # maximum wait for any later frame

# Diagnostic stages (numbering matches the VISENTRA camera diagnostics list).
STAGE_ADB = "1/7 ADB connection failure"
STAGE_SERVER = "2/7 scrcpy server startup failure"
STAGE_SOCKET = "3/7 socket connection failure"
STAGE_NO_BYTES = "4/7 socket connected but no bytes received"
STAGE_NO_FRAMES = "5/7 encoded data received but no frames decoded"
STAGE_CLOSED = "7/7 camera stream closed or interrupted"


class VivoCameraError(RuntimeError):
    """Fatal camera error with the diagnostic stage that failed."""

    def __init__(self, stage, detail):
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def _adb(args, serial=None, timeout=15):
    command = [str(ADB_EXE)]
    if serial:
        command += ["-s", serial]
    command += [str(a) for a in args]
    try:
        return subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise VivoCameraError(
            STAGE_ADB, f"'adb {' '.join(str(a) for a in args)}' timed out after {timeout}s"
        ) from exc
    except OSError as exc:
        raise VivoCameraError(STAGE_ADB, f"could not run {ADB_EXE}: {exc}") from exc


def select_adb_device(requested_serial=None):
    """Return the serial of the one authorized device this session will use."""
    result = _adb(["devices", "-l"], timeout=20)
    if result.returncode != 0:
        raise VivoCameraError(
            STAGE_ADB, f"'adb devices' failed:\n{result.stdout}\n{result.stderr}".strip()
        )

    ready, problems = [], []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == "device":
            ready.append((parts[0], line))
        else:
            problems.append(f"{parts[0]}: {parts[1]}")

    if requested_serial:
        for serial, _ in ready:
            if serial == requested_serial:
                return serial
        raise VivoCameraError(
            STAGE_ADB,
            f"requested device '{requested_serial}' is not an authorized, connected device. "
            f"Ready: {[s for s, _ in ready]}; other: {problems}",
        )

    if not ready:
        raise VivoCameraError(
            STAGE_ADB,
            "no authorized Android device found. Connect the Vivo by USB, enable USB debugging "
            f"and accept the RSA prompt. adb reported: {problems or 'no devices'}",
        )
    if len(ready) == 1:
        return ready[0][0]

    vivo = [item for item in ready if re.search(r"vivo|V2231", item[1], re.IGNORECASE)]
    if len(vivo) == 1:
        return vivo[0][0]
    raise VivoCameraError(
        STAGE_ADB,
        f"several devices are connected {[s for s, _ in ready]}; choose one with --serial <id>.",
    )


def detect_scrcpy_server_version():
    """The server jar must be started with the exact version of the scrcpy client."""
    try:
        completed = subprocess.run(
            [str(SCRCPY_EXE), "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10, cwd=str(ROOT),
        )
        match = re.search(r"scrcpy\s+v?(\d+\.\d+(?:\.\d+)?)", completed.stdout + completed.stderr)
        if match:
            return match.group(1)
    except (OSError, subprocess.TimeoutExpired):
        pass
    log_warn(
        "Could not read the scrcpy version; using "
        f"{SCRCPY_SERVER_VERSION_FALLBACK}."
    )
    return SCRCPY_SERVER_VERSION_FALLBACK


def find_scrcpy_server_file():
    folder = SCRCPY_EXE.parent
    for name in ("scrcpy-server", "scrcpy-server-v5.0.1"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    matches = sorted(folder.glob("scrcpy-server*"))
    for candidate in matches:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the scrcpy server file beside scrcpy.exe. "
        f"Expected scrcpy-server or scrcpy-server-v5.0.1 in {folder}"
    )


def _server_failure_hint(log_text):
    low = log_text.lower()
    if "address already in use" in low:
        return ("another process already owns this scrcpy socket name. VISENTRA uses a unique "
                "socket per session, so look for another program using the same name.")
    if "does not match" in low or ("version" in low and "mismatch" in low):
        return "the scrcpy server version does not match scrcpy.exe; both must be the same release."
    if "securityexception" in low or ("camera" in low and "permission" in low):
        return "Android refused camera access. Check camera permission and any privacy toggle."
    if "cameraaccessexception" in low or "could not open camera" in low or "camera in use" in low:
        return "the camera could not be opened. Close other camera apps and retry."
    if "size" in low and ("not supported" in low or "could not find" in low or "invalid" in low):
        return ("the requested camera size is not supported. Run 'scrcpy.exe --list-camera-sizes' "
                "and pass a listed size with --width/--height.")
    return None


class VivoCameraStream:
    """Decode the raw H.264 camera stream on a background thread.

    Interface kept compatible with cv2.VideoCapture: read() -> (ok, frame) and release().
    Only the newest decoded frame is kept, so a slow consumer never builds a backlog.
    read() waits a bounded time; it never loops forever.
    """

    def __init__(self, sock, initial_data=b""):
        import av

        self.sock = sock
        self.decoder = av.CodecContext.create("h264", "r")
        self._cond = threading.Condition()
        self._latest = None
        self._latest_seq = 0
        self._read_seq = 0
        self._eof = False
        self._reader_error = None
        self._last_read_error = None
        self.closed = False
        self.bytes_received = 0
        self.packets_parsed = 0
        self.frames_decoded = 0
        self.decode_errors = 0
        self._consecutive_decode_errors = 0
        self._initial = initial_data
        self._thread = threading.Thread(
            target=self._reader, daemon=True, name="VISENTRA-VivoReader"
        )
        self._thread.start()

    # ---------------- background reader ----------------
    def _reader(self):
        try:
            self.sock.settimeout(1.0)
            data = self._initial
            self._initial = b""
            first_chunk = True
            while not self.closed:
                if not data:
                    try:
                        data = self.sock.recv(65536)
                    except socket.timeout:
                        data = b""
                        continue
                    except OSError as exc:
                        if not self.closed:
                            self._finish(error=f"socket error: {exc}")
                        return
                    if not data:
                        self._finish(eof=True)
                        return
                self.bytes_received += len(data)
                if first_chunk:
                    first_chunk = False
                    if data.find(b"\x00\x00\x01", 0, 64) < 0:
                        log_warn(
                            "The first bytes do not look like raw Annex-B H.264 "
                            f"({data[:16].hex()}). If no frames follow, the server may be "
                            "sending scrcpy framing headers instead of a raw stream."
                        )
                self._feed(data)
                data = b""
        except Exception as exc:  # noqa: BLE001 - report the real cause to the consumer
            self._finish(error=f"decoder thread failed: {exc!r}")

    def _feed(self, data):
        try:
            packets = self.decoder.parse(data)
        except Exception as exc:  # noqa: BLE001
            self._note_decode_error(f"H.264 parse error: {exc!r}")
            return
        for packet in packets:
            self.packets_parsed += 1
            try:
                frames = self.decoder.decode(packet)
            except Exception as exc:  # noqa: BLE001
                self._note_decode_error(f"H.264 decode error: {exc!r}")
                continue
            for frame in frames:
                try:
                    array = frame.to_ndarray(format="bgr24")
                except Exception as exc:  # noqa: BLE001
                    self._note_decode_error(f"frame conversion error: {exc!r}")
                    continue
                if (array is None or array.ndim != 3 or array.shape[0] < 2
                        or array.shape[1] < 2 or array.shape[2] != 3):
                    self._note_decode_error("decoded frame has invalid dimensions")
                    continue
                self._consecutive_decode_errors = 0
                self.frames_decoded += 1
                with self._cond:
                    self._latest = array
                    self._latest_seq += 1
                    self._cond.notify_all()

    def _note_decode_error(self, message):
        self.decode_errors += 1
        self._consecutive_decode_errors += 1
        if self.decode_errors <= 5:
            log_warn(message)
        if self._consecutive_decode_errors >= 300:
            self._finish(error=f"{message} (300 consecutive failures)")

    def _finish(self, eof=False, error=None):
        with self._cond:
            if eof:
                self._eof = True
            if error and self._reader_error is None:
                self._reader_error = error
            self._cond.notify_all()

    # ---------------- consumer API ----------------
    @property
    def failed(self):
        return self._eof or self._reader_error is not None or self.closed

    def read(self, timeout=VIVO_READ_TIMEOUT_S):
        """Return (True, newest unseen BGR frame) or (False, None) after a bounded wait."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest_seq == self._read_seq:
                if self.closed or self._eof or self._reader_error is not None:
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._last_read_error = f"no new frame within {timeout:.1f}s"
                    return False, None
                self._cond.wait(remaining)
            self._read_seq = self._latest_seq
            return True, self._latest

    def stats(self):
        return (f"{self.bytes_received} bytes, {self.packets_parsed} packets, "
                f"{self.frames_decoded} frames, {self.decode_errors} decode errors")

    def diagnose(self):
        """Return (stage, detail) describing why read() gave no frame."""
        if self._reader_error:
            reason = self._reader_error
        elif self._eof:
            reason = "the device closed the stream"
        else:
            reason = self._last_read_error or "no frame available"
        if self.bytes_received == 0:
            return STAGE_NO_BYTES, f"{reason}; nothing was received ({self.stats()})"
        if self.frames_decoded == 0:
            return STAGE_NO_FRAMES, f"{reason}; {self.stats()}"
        return STAGE_CLOSED, f"{reason}; {self.stats()}"

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        with self._cond:
            self._cond.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)


_VIVO_ACTIVE_LOCK = threading.Lock()
_VIVO_ACTIVE_SESSION = None


class VivoCameraSession:
    """Owns everything started for one live session: jar, forward, server, socket."""

    def __init__(self, serial, server_version, width, height, fps):
        self.serial = serial
        self.server_version = server_version
        self.width, self.height, self.fps = width, height, fps
        # Unique socket per session: another scrcpy run (or a leftover server from an
        # earlier crash) can never collide with "Address already in use".
        self.scid = random.randint(1, 0x7FFFFFFF)
        self.socket_name = f"scrcpy_{self.scid:08x}"
        self.local_port = None
        self.server_process = None
        self.stream = None
        self._log = deque(maxlen=150)
        self._log_thread = None
        self._closed = False
        self._pushed = False

    def log_text(self, last=30):
        return "\n".join(list(self._log)[-last:])

    def _drain_server_output(self):
        try:
            for line in self.server_process.stdout:
                line = line.rstrip()
                if line:
                    self._log.append(line)
        except (OSError, ValueError):
            pass

    # ---------------- startup ----------------
    def start(self, server_file):
        push = _adb(["push", str(server_file), VIVO_REMOTE_SERVER_PATH], self.serial, timeout=60)
        if push.returncode != 0:
            raise VivoCameraError(STAGE_ADB, f"could not push the scrcpy server:\n{push.stderr}")
        self._pushed = True

        self.local_port = self._create_forward()

        command = [
            str(ADB_EXE), "-s", self.serial, "shell",
            f"CLASSPATH={VIVO_REMOTE_SERVER_PATH}",
            "app_process", "/", "com.genymobile.scrcpy.Server", self.server_version,
            f"scid={self.scid:08x}",
            "tunnel_forward=true",
            "audio=false",
            "control=false",
            "cleanup=false",
            "raw_stream=true",
            "video_codec=h264",
            "video_source=camera",
            "camera_facing=back",
            f"camera_size={self.width}x{self.height}",
            f"camera_fps={self.fps}",
            "log_level=debug",
        ]
        log_info(
            f"Starting Vivo rear camera H.264 stream (device {self.serial}, "
            f"scrcpy server {self.server_version}, socket {self.socket_name}, "
            f"local port {self.local_port})..."
        )
        try:
            self.server_process = subprocess.Popen(
                command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as exc:
            raise VivoCameraError(STAGE_SERVER, f"could not launch adb shell: {exc}") from exc
        self._log_thread = threading.Thread(
            target=self._drain_server_output, daemon=True, name="VISENTRA-ScrcpyLog"
        )
        self._log_thread.start()

    def _create_forward(self):
        last_error = ""
        for _ in range(5):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
            result = _adb(
                ["forward", f"tcp:{port}", f"localabstract:{self.socket_name}"], self.serial
            )
            if result.returncode == 0:
                return port
            last_error = result.stderr.strip() or result.stdout.strip()
        raise VivoCameraError(STAGE_ADB, f"ADB port forwarding failed: {last_error}")

    # ---------------- connection ----------------
    def _server_failure_text(self):
        code = self.server_process.returncode
        tail = self.log_text(25)
        text = f"the scrcpy camera server exited with code {code}."
        if tail:
            text += f"\n--- last server output ---\n{tail}\n--- end of server output ---"
        hint = _server_failure_hint(tail)
        if hint:
            text += f"\nHint: {hint}"
        return text

    def _wait_for_first_bytes(self, sock, deadline):
        sock.settimeout(1.0)
        while time.monotonic() < deadline:
            if self.server_process.poll() is not None:
                return "server_exit", b""
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return "closed", b""
            if not data:
                return "closed", b""
            return "data", data
        return "timeout", b""

    def connect(self, timeout=VIVO_CONNECT_TIMEOUT_S):
        """Connect and return a stream only after real video bytes have arrived.

        adb accepts the local TCP connection even when nothing listens on the phone yet and
        then closes it, so a successful connect() proves nothing; bytes do.
        """
        deadline = time.monotonic() + timeout
        outcome = "never connected"
        closed_early = 0
        last_connect_error = None
        while time.monotonic() < deadline:
            if self.server_process.poll() is not None:
                time.sleep(0.3)  # let the log thread collect the final lines
                raise VivoCameraError(STAGE_SERVER, self._server_failure_text())
            try:
                sock = socket.create_connection(("127.0.0.1", self.local_port), timeout=2)
            except OSError as exc:
                last_connect_error = exc
                outcome = "connect_failed"
                time.sleep(0.25)
                continue

            status, data = self._wait_for_first_bytes(sock, deadline)
            if status == "data":
                return VivoCameraStream(sock, data)
            sock.close()
            if status == "server_exit":
                time.sleep(0.3)
                raise VivoCameraError(STAGE_SERVER, self._server_failure_text())
            if status == "closed":
                closed_early += 1
                outcome = "closed_early"
                time.sleep(0.25)
                continue
            outcome = "no_bytes"
            break

        if self.server_process.poll() is not None:
            raise VivoCameraError(STAGE_SERVER, self._server_failure_text())
        tail = self.log_text(15)
        extra = f"\n--- last server output ---\n{tail}\n--- end ---" if tail else ""
        if outcome == "no_bytes":
            raise VivoCameraError(
                STAGE_NO_BYTES,
                f"connected to 127.0.0.1:{self.local_port} but no video bytes arrived within "
                f"{timeout:.0f}s (the server is running but is not streaming).{extra}",
            )
        if outcome == "closed_early":
            raise VivoCameraError(
                STAGE_SOCKET,
                f"adb accepted the connection {closed_early} time(s) but closed it before any "
                f"bytes: the server never started listening on localabstract:{self.socket_name}."
                f"{extra}",
            )
        raise VivoCameraError(
            STAGE_SOCKET,
            f"could not connect to 127.0.0.1:{self.local_port}: {last_connect_error}{extra}",
        )

    # ---------------- cleanup ----------------
    def close(self):
        """Idempotent; stops only what this session started."""
        global _VIVO_ACTIVE_SESSION
        if self._closed:
            return
        self._closed = True

        if self.stream is not None:
            try:
                self.stream.release()
            except Exception:  # noqa: BLE001
                pass

        process = self.server_process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:  # noqa: BLE001
                pass
        if self._log_thread is not None:
            self._log_thread.join(timeout=1)

        # Stop a still-running on-device server that belongs to THIS session only
        # (the pattern contains our unique scid), then remove our forward and jar.
        if self.serial:
            for args in (
                ["shell", "pkill", "-f", f"scid={self.scid:08x}"],
                ["shell", "rm", "-f", VIVO_REMOTE_SERVER_PATH] if self._pushed else None,
            ):
                if args is None:
                    continue
                try:
                    _adb(args, self.serial, timeout=8)
                except Exception:  # noqa: BLE001
                    pass
            if self.local_port is not None:
                try:
                    _adb(["forward", "--remove", f"tcp:{self.local_port}"], self.serial, timeout=8)
                except Exception:  # noqa: BLE001
                    pass

        with _VIVO_ACTIVE_LOCK:
            if _VIVO_ACTIVE_SESSION is self:
                _VIVO_ACTIVE_SESSION = None


def start_vivo_camera(width=640, height=480, fps=15, serial=None):
    """Start exactly one Vivo camera stream; return (session, stream).

    `stream` offers read()/release() like cv2.VideoCapture. `session.close()` cleans up the
    scrcpy server, adb forward and socket. A valid decoded frame is received before return.
    """
    global _VIVO_ACTIVE_SESSION

    if not ADB_EXE.is_file():
        raise FileNotFoundError(f"ADB executable not found: {ADB_EXE}")
    if not SCRCPY_EXE.is_file():
        raise FileNotFoundError(f"scrcpy executable not found: {SCRCPY_EXE}")
    try:
        import av  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is missing from the Python environment running visentra.py. "
            "Install PyAV in that same environment before starting live mode."
        ) from exc
    server_file = find_scrcpy_server_file()

    with _VIVO_ACTIVE_LOCK:
        if _VIVO_ACTIVE_SESSION is not None:
            raise RuntimeError(
                "A Vivo camera session is already active in this process; "
                "start_vivo_camera() must be called only once per live session."
            )
        serial = select_adb_device(serial)
        session = VivoCameraSession(serial, detect_scrcpy_server_version(), width, height, fps)
        _VIVO_ACTIVE_SESSION = session

    try:
        session.start(server_file)
        stream = session.connect()
        session.stream = stream
        log_info(f"Video bytes are arriving from the Vivo ({serial}); waiting for the first decoded frame...")

        ok, frame = stream.read(timeout=VIVO_FIRST_FRAME_TIMEOUT_S)
        if not ok or frame is None:
            if session.server_process.poll() is not None:
                raise VivoCameraError(STAGE_SERVER, session._server_failure_text())
            stage, detail = stream.diagnose()
            raise VivoCameraError(stage, detail)
        frame_h, frame_w = frame.shape[:2]
        log_info(
            f"6/7 Frames decoded successfully: first frame {frame_w}x{frame_h} "
            f"(requested {width}x{height} at {fps} FPS)."
        )
        if (frame_w, frame_h) != (width, height):
            log_warn(
                f"The device delivers {frame_w}x{frame_h}, not the requested {width}x{height}; "
                "the stream size is used as-is."
            )
        return session, stream
    except BaseException:
        session.close()
        raise


def check_paths(require_lane=True, require_camera=False):
    paths = {
        "RF-DETR Python": RF_PYTHON,
        "RF-DETR model": RF_MODEL,
        "Metric3D Python": METRIC_PYTHON,
        "Metric3D ONNX model": METRIC_MODEL,
        "Metric3D preprocessing config": METRIC_CONFIG,
    }
    if require_lane:
        paths.update({
            "CurveLaneS Python": LANE_PYTHON,
            "CurveLaneS model": LANE_MODEL,
            "CurveLaneS config": LANE_CONFIG,
            "CondLaneNet source": COND_LANE_ROOT,
        })
    if require_camera:
        paths.update({
            "ADB executable": ADB_EXE,
            "scrcpy executable": SCRCPY_EXE,
        })
    print("\n========== VISENTRA PATH CHECK ==========")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        print(f"[OK] {name}: {path}")
    print("=========================================\n")


# ============================================================
# RF-DETR VEHICLE / OBJECT DETECTION
# ============================================================
def get_rf_detector():
    """Load the RF-DETR checkpoint once and reuse it."""
    global _RF_DETECTOR
    if _RF_DETECTOR is None:
        print("[RF-DETR] Loading checkpoint once...")
        from rfdetr import RFDETRBase
        _RF_DETECTOR = RFDETRBase.from_checkpoint(str(RF_MODEL))
        print("[RF-DETR] Model loaded.")
    return _RF_DETECTOR


def run_rfdetr(image_path):
    """Run RF-DETR and return JSON-safe boxes, classes, names and confidences."""
    detector = get_rf_detector()
    detections = detector.predict(str(image_path), threshold=0.40)
    result = {"boxes": [], "class_ids": [], "class_names": [], "confidences": []}
    names = detections.data.get("class_name") if getattr(detections, "data", None) else None

    for index, (box, class_id, confidence) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence
    )):
        result["boxes"].append([float(v) for v in box])
        try:
            result["class_ids"].append(int(class_id))
        except (TypeError, ValueError):
            result["class_ids"].append(-1)
        result["confidences"].append(float(confidence))
        if names is not None and index < len(names):
            result["class_names"].append(str(names[index]))
        else:
            result["class_names"].append(str(class_id))
    return result

# ============================================================
# MOBILENETV3 SMALL - OPTIONAL CROP CLASSIFICATION
# ============================================================
def load_mobilenetv3():
    global _MOBILENETV3_MODEL
    global _MOBILENETV3_TRANSFORM
    global _MOBILENETV3_LABELS

    if _MOBILENETV3_MODEL is not None:
        return

    import torch
    from torchvision.models import (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights,
    )

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    model.cpu()

    _MOBILENETV3_MODEL = model
    _MOBILENETV3_TRANSFORM = weights.transforms()
    _MOBILENETV3_LABELS = weights.meta["categories"]

    print("[MobileNetV3] Pretrained Small model loaded on CPU.")


def classify_mobilenetv3_crops(image_path, detections):
    """
    Classify detected vehicle crops with MobileNetV3 Small.

    Returns one result per detection. ImageNet labels are generic categories;
    they should not be interpreted as reliable vehicle make/model predictions.
    """
    results = [
        {"label": None, "confidence": None}
        for _ in detections.get("boxes", [])
    ]

    if not MOBILENETV3_ENABLED:
        return results

    try:
        load_mobilenetv3()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        vehicle_count = 0

        for index, box in enumerate(detections.get("boxes", [])):
            class_names = detections.get("class_names", [])
            if index >= len(class_names):
                continue
            if not is_vehicle_class(class_names[index]):
                continue

            if vehicle_count >= MOBILENETV3_MAX_CROPS:
                break
            vehicle_count += 1

            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            tensor = _MOBILENETV3_TRANSFORM(crop).unsqueeze(0)

            # inference_mode is self-contained here: only Python floats leave the block.
            with torch.inference_mode():
                logits = _MOBILENETV3_MODEL(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                confidence, class_index = torch.max(probabilities, dim=0)
                label_index = int(class_index)
                confidence_value = float(confidence)

            results[index] = {
                "label": _MOBILENETV3_LABELS[label_index],
                "confidence": confidence_value,
            }

    except Exception as exc:
        # Classification is optional; a failure must not stop RF-DETR.
        print(f"[MobileNetV3] Classification unavailable: {exc}",
              file=sys.stderr)

    return results


def attach_mobilenetv3_results(image_path, detections):
    detections = dict(detections)
    detections["mobilenetv3"] = classify_mobilenetv3_crops(
        image_path, detections
    )
    return detections


# ============================================================
# CURVELANES MODEL LOADING
# ============================================================
def load_curvelanes():
    print("[CurveLaneS] Loading checkpoint once...")
    cond_root = str(COND_LANE_ROOT)
    if cond_root not in sys.path:
        sys.path.insert(0, cond_root)

    import torch
    if not torch.cuda.is_available():
        # Compatibility patch retained for this CPU-only environment.
        torch.cuda.FloatTensor = torch.FloatTensor

    import mmcv
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    config_namespace = {"__file__": str(LANE_CONFIG), "__name__": "__curvelanes_config__"}
    with open(LANE_CONFIG, "r", encoding="utf-8") as handle:
        config_source = handle.read()
    exec(compile(config_source, str(LANE_CONFIG), "exec"), config_namespace)
    cfg_dict = {key: value for key, value in config_namespace.items() if not key.startswith("__")}
    cfg = mmcv.Config(cfg_dict)
    cfg.model.head.location_configs.device = "cpu"
    cfg.model.pretrained = None

    model = build_detector(cfg.model)
    load_checkpoint(model, str(LANE_MODEL), map_location="cpu")
    model = model.cpu()
    model.eval()
    print("[CurveLaneS] Model loaded.")
    return model, cfg


def prepare_curvelanes_image(image_path, cfg):
    global _LANE_PIPELINE
    from mmdet.datasets.pipelines import Compose
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_shape = image.shape
    results = {
        "filename": str(image_path),
        "sub_img_name": Path(image_path).name,
        "img": image,
        "gt_points": [],
        "id_classes": [],
        "id_instances": [],
        "img_shape": original_shape,
        "ori_shape": original_shape,
        "crop_offset": [0, 0],
        "crop_shape": original_shape,
    }
    # Construct the validation pipeline once per worker instead of once per frame.
    # These transforms are used as configured by the existing checkpoint/config.
    if _LANE_PIPELINE is None:
        _LANE_PIPELINE = Compose(cfg.val_pipeline)
    data = _LANE_PIPELINE(results)
    if data is None:
        raise RuntimeError("CurveLaneS preprocessing returned None.")
    return data, original_shape


def unwrap_data_container(value):
    if not hasattr(value, "data"):
        return value
    value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def prepare_model_inputs(data):
    import torch
    img = unwrap_data_container(data["img"])
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    img = img.float()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() != 4:
        raise RuntimeError(f"Unexpected CurveLaneS image tensor shape: {tuple(img.shape)}")
    img = img.cpu()

    meta_container = data["img_metas"]
    meta = meta_container.data if hasattr(meta_container, "data") else meta_container
    while isinstance(meta, (list, tuple)) and len(meta) == 1:
        meta = meta[0]
    if isinstance(meta, (list, tuple)):
        if not meta:
            raise RuntimeError("CurveLaneS img_metas is empty.")
        if isinstance(meta[0], dict):
            meta = meta[0]
    if not isinstance(meta, dict):
        raise RuntimeError(f"Could not extract CurveLaneS metadata; got {type(meta).__name__}")
    return img, [meta], meta


# ============================================================
# CURVELANES INFERENCE
# ============================================================
def get_lane_detector():
    """Load the CurveLaneS model once and reuse it."""
    global _LANE_DETECTOR, _LANE_CFG
    if _LANE_DETECTOR is None:
        _LANE_DETECTOR, _LANE_CFG = load_curvelanes()
    return _LANE_DETECTOR, _LANE_CFG


def run_curvelanes(image_path):
    global _LANE_POSTPROCESSOR
    cond_root = str(COND_LANE_ROOT)
    if cond_root not in sys.path:
        sys.path.insert(0, cond_root)
    from tools.condlanenet.post_process import CurvelanesPostProcessor

    model, cfg = get_lane_detector()

    data, original_shape = prepare_curvelanes_image(image_path, cfg)
    img, img_metas, meta = prepare_model_inputs(data)
    import torch
    # torch.no_grad() is kept (not inference_mode): the post-processor below touches the seed
    # tensors outside this block, and inference tensors can reject such operations.
    with torch.no_grad():
        seeds, _ = model(
            img=img,
            img_metas=img_metas,
            return_loss=False,
            rescale=False,
            thr=0.3,
        )

    # Reuse the stateless postprocessor; recreating it every frame adds avoidable overhead.
    if _LANE_POSTPROCESSOR is None:
        _LANE_POSTPROCESSOR = CurvelanesPostProcessor(mask_size=cfg.mask_size, hm_thr=0.3)
    lanes, _ = _LANE_POSTPROCESSOR(seeds, meta["down_scale"])
    processed_shape = meta["img_shape"]
    h_processed, w_processed = processed_shape[0], processed_shape[1]
    h_original, w_original = original_shape[0], original_shape[1]
    ratio_x = w_original / w_processed
    ratio_y = h_original / h_processed

    final_lanes = []
    if lanes is not None:
        for lane in lanes:
            points = []
            for point in lane.get("points", []):
                x = max(0.0, min(float(point[0]) * ratio_x, w_original - 1))
                y = max(0.0, min(float(point[1]) * ratio_y, h_original - 1))
                points.append([int(round(x)), int(round(y))])
            if len(points) > 1:
                final_lanes.append(points)
    return final_lanes


# ============================================================
# METRIC3D V2 DEPTH ESTIMATION
# ============================================================
class Metric3DEstimator:
    """Persistent CPU ONNX Runtime wrapper for the downloaded Metric3D v2 export."""

    def __init__(self):
        import onnxruntime as ort
        from PIL import Image  # noqa: F401 - validates dependency early

        if not METRIC_MODEL.is_file():
            raise FileNotFoundError(f"Metric3D model not found: {METRIC_MODEL}")
        if not METRIC_CONFIG.is_file():
            raise FileNotFoundError(f"Metric3D config not found: {METRIC_CONFIG}")

        with METRIC_CONFIG.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(os.environ.get("VISENTRA_ORT_THREADS", "3")))
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            # Idle ORT threads busy-wait by default; with three CPU models sharing one laptop
            # CPU that spinning steals time from the other workers.
            session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:  # noqa: BLE001 - older ORT builds may not know this key
            pass
        self.session = ort.InferenceSession(
            str(METRIC_MODEL),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        names = [item.name.lower() for item in self.session.get_outputs()]
        self.depth_output_index = next(
            (index for index, name in enumerate(names) if "depth" in name), None
        )
        if self.depth_output_index is None:
            raise RuntimeError(f"Could not find depth output in: {names}")
        print("[Metric3D] ONNX model loaded on CPU.")
        print(f"[Metric3D] Input: {self.input_name}; outputs: {names}")

    def predict_with_size(self, image_path):
        """Return (depth_map, (original_height, original_width)); the image is decoded once."""
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise RuntimeError(f"Invalid image dimensions: {image_path}")

        config = self.config
        target_h = int(config.get("size", {}).get("height", 518))
        target_w = int(config.get("size", {}).get("width", 518))
        divisor = max(1, int(config.get("ensure_multiple_of", 14)))

        if config.get("do_resize", True):
            if config.get("keep_aspect_ratio", True):
                scale = min(target_w / original_width, target_h / original_height)
                resized_w = max(
                    divisor,
                    min(target_w, int(round(original_width * scale / divisor)) * divisor),
                )
                resized_h = max(
                    divisor,
                    min(target_h, int(round(original_height * scale / divisor)) * divisor),
                )
            else:
                resized_w, resized_h = target_w, target_h
            image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        rgb = np.asarray(image, dtype=np.float32)
        # Match the downloaded preprocessor_config.json: no rescale or normalization.
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: tensor})
        depth = np.squeeze(np.asarray(outputs[self.depth_output_index], dtype=np.float32))
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected Metric3D depth tensor shape: {depth.shape}")
        if not np.isfinite(depth).any():
            raise RuntimeError("Metric3D depth output contains no finite values.")
        return depth, (original_height, original_width)

    def predict(self, image_path):
        depth, _ = self.predict_with_size(image_path)
        return depth


def estimate_box_distances(detections, depth_map, image_shape, depth_scale):
    """Sample central box regions from the full-image depth map; retain box ordering."""
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    depth_h, depth_w = depth_map.shape[:2]
    boxes = detections.get("boxes", []) or []
    class_ids = detections.get("class_ids", []) or []
    class_names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []
    distance_results = []

    for index, box in enumerate(boxes):
        item = {
            "detection_index": int(index),
            "class_id": class_ids[index] if index < len(class_ids) else -1,
            "class_name": class_names[index] if index < len(class_names) else "object",
            "confidence": float(confidences[index]) if index < len(confidences) else None,
            "box_xyxy": [float(value) for value in box],
            "distance_m": None,
            "raw_depth_median": None,
            "raw_depth_p25": None,
            "raw_depth_p75": None,
            "depth_spread_m": None,
            "valid_pixel_fraction": 0.0,
            "capped_pixel_fraction": 0.0,
            "sample_pixels": 0,
            "status": "invalid_box",
        }
        # Keep detections such as people/benches available for drawing, but
        # never calculate or publish them as vehicle distances.
        if not is_vehicle_class(item["class_name"]):
            item["status"] = "non_vehicle_skipped"
            distance_results.append(item)
            continue

        if len(box) < 4:
            distance_results.append(item)
            continue

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0.0, min(x1, image_w - 1.0))
        x2 = max(0.0, min(x2, image_w - 1.0))
        y1 = max(0.0, min(y1, image_h - 1.0))
        y2 = max(0.0, min(y2, image_h - 1.0))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 2 or box_h < 2:
            distance_results.append(item)
            continue

        # Sample the interior of the detection to reduce edge/background contamination.
        # If this interior is too small, fall back to the full detection box.
        def crop_to_depth(xa, ya, xb, yb):
            dx1 = max(0, min(depth_w - 1, int(np.floor(xa * depth_w / image_w))))
            dy1 = max(0, min(depth_h - 1, int(np.floor(ya * depth_h / image_h))))
            dx2 = max(dx1 + 1, min(depth_w, int(np.ceil(xb * depth_w / image_w))))
            dy2 = max(dy1 + 1, min(depth_h, int(np.ceil(yb * depth_h / image_h))))
            return depth_map[dy1:dy2, dx1:dx2]

        inner = crop_to_depth(
            x1 + 0.20 * box_w,
            y1 + 0.20 * box_h,
            x2 - 0.20 * box_w,
            y2 - 0.10 * box_h,
        )
        if inner.size < 4:
            inner = crop_to_depth(x1, y1, x2, y2)

        finite = np.isfinite(inner)
        capped_fraction = float(np.mean(finite & (inner >= DEPTH_CAP_VALUE))) if inner.size else 0.0
        valid_mask = finite & (inner > 0.0) & (inner < DEPTH_CAP_VALUE)
        values = inner[valid_mask]
        item["sample_pixels"] = int(inner.size)
        item["valid_pixel_fraction"] = float(values.size / inner.size) if inner.size else 0.0
        item["capped_pixel_fraction"] = capped_fraction

        if values.size < 2:
            item["status"] = "insufficient_uncapped_depth_pixels"
            distance_results.append(item)
            continue

        p25, median, p75 = [float(v) for v in np.percentile(values, [25, 50, 75])]
        item.update({
            "raw_depth_median": median,
            "raw_depth_p25": p25,
            "raw_depth_p75": p75,
            "distance_m": median * float(depth_scale),
            "depth_spread_m": (p75 - p25) * float(depth_scale),
            "status": "experimental_estimate",
        })
        distance_results.append(item)

    return distance_results


def get_metric_estimator():
    global _METRIC_ESTIMATOR
    if _METRIC_ESTIMATOR is None:
        _METRIC_ESTIMATOR = Metric3DEstimator()
    return _METRIC_ESTIMATOR


def run_metric3d(image_path, detections, depth_scale, save_artifacts=False):
    estimator = get_metric_estimator()
    # One image decode: predict_with_size also returns the original size that
    # estimate_box_distances needs (previously the image was decoded a second time with cv2).
    depth_map, image_size = estimator.predict_with_size(image_path)
    distances = estimate_box_distances(detections, depth_map, image_size, depth_scale)

    result = {
        "distances": distances,
        "depth_shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        "depth_scale": float(depth_scale),
        "calibration_status": "provisional_not_independently_validated",
    }
    if save_artifacts:
        stem = Path(image_path).stem
        raw_path = METRIC_RESULTS_DIR / f"visentra_{stem}_raw_depth.npy"
        viz_path = METRIC_RESULTS_DIR / f"visentra_{stem}_depth_visualization.png"
        np.save(raw_path, depth_map)
        finite = np.isfinite(depth_map)
        display = np.zeros(depth_map.shape, dtype=np.uint8)
        if finite.any():
            lo, hi = np.percentile(depth_map[finite], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            display[finite] = np.clip(
                (depth_map[finite] - lo) * 255.0 / (hi - lo), 0, 255
            ).astype(np.uint8)
        heatmap = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(viz_path), heatmap)
        result["raw_depth_path"] = str(raw_path)
        result["depth_visualization_path"] = str(viz_path)
    return result


# ============================================================
# DRAWING
# ============================================================
def select_two_lane_boundaries(lanes, image_shape):
    """Display heuristic: choose closest predicted curve on either side of centre."""
    height, width = image_shape[:2]
    centre_x = width / 2.0
    candidates = []
    for lane in lanes or []:
        if len(lane) < 2:
            continue
        lower = [point for point in lane if point[1] >= height * 0.55]
        points = lower if lower else lane
        representative_x = float(np.mean([point[0] for point in points]))
        candidates.append((representative_x, lane))

    left = [item for item in candidates if item[0] < centre_x]
    right = [item for item in candidates if item[0] >= centre_x]
    selected = []
    if left:
        selected.append(max(left, key=lambda item: item[0])[1])
    if right:
        selected.append(min(right, key=lambda item: item[0])[1])
    return selected


def update_vehicle_tracks(detections, tracker):
    """Update ByteTrack once and preserve original detection ordering."""
    import supervision as sv

    boxes = detections.get("boxes", []) or []
    class_ids = detections.get("class_ids", []) or []
    class_names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []

    result = dict(detections)
    result["track_ids"] = [None] * len(boxes)

    # Track supported vehicle classes only.
    source_indices = [
        i for i, _ in enumerate(boxes)
        if i < len(class_names) and is_vehicle_class(class_names[i])
        and i < len(confidences)
        and i < len(class_ids)
        and len(boxes[i]) == 4
    ]

    xyxy = np.asarray(
        [boxes[i] for i in source_indices], dtype=np.float32
    ).reshape(-1, 4)

    confidence = np.asarray(
        [confidences[i] for i in source_indices], dtype=np.float32
    )

    classes = np.asarray(
        [class_ids[i] for i in source_indices], dtype=np.int32
    )

    vehicle_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=classes,
        data={"source_index": np.asarray(source_indices, dtype=np.int32)},
    )

    tracked = tracker.update_with_detections(vehicle_detections)

    if tracked.tracker_id is not None:
        # Match tracked boxes back to the original detection indexes.
        # Do not rely on custom fields surviving tracker.update_with_detections().
        tracked_boxes = np.asarray(tracked.xyxy, dtype=np.float32).reshape(-1, 4)
        tracked_ids = tracked.tracker_id
        original_boxes = xyxy
        used_source_indices = set()

        for tracked_box, track_id in zip(tracked_boxes, tracked_ids):
            if track_id is None:
                continue

            best_index = None
            best_iou = 0.0

            for local_index, original_box in enumerate(original_boxes):
                source_index = source_indices[local_index]
                if source_index in used_source_indices:
                    continue

                ax1, ay1, ax2, ay2 = tracked_box
                bx1, by1, bx2, by2 = original_box

                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)

                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_index = source_index

            if best_index is not None and best_iou >= 0.20:
                result["track_ids"][best_index] = int(track_id)
                used_source_indices.add(best_index)

    return result

# ============================================================
# DEEPSORT - OPTIONAL SECOND TRACKER FOR COMPARISON WITH BYTETRACK (--with-deepsort)
# ============================================================
def update_deepsort_tracks(frame, detections, tracker):
    """
    Run DeepSORT on vehicle detections and map its IDs to detection indexes.
    ByteTrack remains separate and continues running independently.
    """
    boxes = detections.get("boxes", []) or []
    names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []

    result = dict(detections)
    result["deepsort_ids"] = [None] * len(boxes)

    raw_detections = []
    source_indices = []

    for index, box in enumerate(boxes):
        if index >= len(names) or index >= len(confidences):
            continue
        if not is_vehicle_class(names[index]) or len(box) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        left = max(0.0, x1)
        top = max(0.0, y1)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        if width < 2 or height < 2:
            continue

        raw_detections.append((
            [left, top, width, height],
            float(confidences[index]),
            str(names[index]),
        ))
        source_indices.append(index)

    # With no detections, pass empty embeddings so tracks still age without running the embedder.
    tracks = tracker.update_tracks(
        raw_detections, embeds=[] if not raw_detections else None, frame=frame
    )

    # Match each confirmed track to the current detection by IoU.
    # This mapping is approximate when boxes overlap heavily.
    candidates = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        try:
            x1, y1, x2, y2 = map(float, track.to_ltrb())
        except (TypeError, ValueError):
            continue

        candidates.append((str(track.track_id), [x1, y1, x2, y2]))

    used_sources = set()

    for track_id, track_box in candidates:
        best_source = None
        best_iou = 0.0

        for source_index in source_indices:
            if source_index in used_sources:
                continue

            original_box = [float(v) for v in boxes[source_index][:4]]
            ax1, ay1, ax2, ay2 = track_box
            bx1, by1, bx2, by2 = original_box

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)

            intersection = (
                max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            )
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_source = source_index

        if best_source is not None and best_iou >= 0.20:
            result["deepsort_ids"][best_source] = track_id
            used_sources.add(best_source)

    return result



def draw_results(
    image_or_path,
    detections,
    lanes,
    output_path=None,
    fps=None,
    frame_number=None,
    distances=None,
    depth_scale=DEFAULT_DEPTH_SCALE,
    status_lines=None,
    lanes_enabled=True,
):
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path))
    else:
        image = image_or_path.copy()
    if image is None:
        raise RuntimeError(f"Could not read image: {image_or_path}")

    vehicle_color = (0, 180, 0)
    lane_colors = [(0, 220, 255), (255, 160, 0)]
    boxes = detections.get("boxes", [])
    class_ids = detections.get("class_ids", [])
    class_names = detections.get("class_names", [])
    confidences = detections.get("confidences", [])
    distance_by_index = {}
    if distances is not None:
        distance_by_index = {
            int(entry.get("detection_index", -1)): entry
            for entry in distances
        }

    for index, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue


        label = class_names[index] if index < len(class_names) else f"class {class_id}"

        # Show ByteTrack and DeepSORT IDs separately.
        track_ids = detections.get("track_ids", [])
        if index < len(track_ids) and track_ids[index] is not None:
            label = f"BT {track_ids[index]} | {label}"

        deepsort_ids = detections.get("deepsort_ids", [])
        if index < len(deepsort_ids) and deepsort_ids[index] is not None:
            label = f"{label} | DS {deepsort_ids[index]}"

        # Show the optional MobileNetV3 ImageNet prediction, if available.
        mobilenet_results = detections.get("mobilenetv3", [])
        if index < len(mobilenet_results):
            classification = mobilenet_results[index]
            if classification and classification.get("label"):
                label += (
                    f" | MN: {classification['label']}"
                    f" {classification['confidence']:.2f}"
                )

        distance_entry = distance_by_index.get(index)
        original_class_name = (
            class_names[index] if index < len(class_names) else ""
        )

        if is_vehicle_class(original_class_name):
            if distance_entry and distance_entry.get("distance_m") is not None:
                label += f" | ~{float(distance_entry['distance_m']):.1f} m"
                if distance_entry.get("stale"):
                    label += "*"  # carried over from an earlier frame for the same track
            elif distance_entry and distance_entry.get("status") == "pending":
                label += " | distance pending"
            elif distances is not None:
                label += " | distance unavailable"
        cv2.rectangle(image, (x1, y1), (x2, y2), vehicle_color, 2)
        cv2.putText(
            image, label[:100], (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, vehicle_color, 2, cv2.LINE_AA,
        )

    image_centre_x = image.shape[1] / 2.0
    for lane in lanes or []:
        if len(lane) < 2:
            continue
        lower_points = [point for point in lane if point[1] >= image.shape[0] * 0.55]
        representative_points = lower_points if lower_points else lane
        representative_x = float(np.mean([point[0] for point in representative_points]))
        is_left = representative_x < image_centre_x
        color = lane_colors[0] if is_left else lane_colors[1]
        points = np.asarray(lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, color, 5, cv2.LINE_AA)
        label_x, label_y = lane[len(lane) // 2]
        cv2.putText(
            image, "Left boundary" if is_left else "Right boundary",
            (int(label_x), max(24, int(label_y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    status = (f"Lane boundaries: {len(lanes or [])}/2" if lanes_enabled
              else "Lane detection: off (use --with-lanes)")
    if fps is not None:
        status += f" | display FPS: {fps:.2f}"
    if frame_number is not None:
        status += f" | frame: {frame_number}"
    cv2.putText(image, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA)
    lines = list(status_lines or [])
    lines.append(f"Metric3D experimental scale: {float(depth_scale):.6f} (not safety-validated)")
    for offset, text in enumerate(lines, start=1):
        cv2.putText(image, str(text)[:140], (12, 28 + offset * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return image


# ============================================================
# SINGLE-IMAGE WORKERS
# ============================================================
def worker_rf(image_path, result_path):
    result = run_rfdetr(image_path)
    result = attach_mobilenetv3_results(image_path, result)

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

def worker_lane(image_path, result_path):
    result = {"lanes": run_curvelanes(image_path)}
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def worker_metric(image_path, detections_path, result_path, depth_scale):
    with open(detections_path, "r", encoding="utf-8") as handle:
        detections = json.load(handle)
    result = run_metric3d(image_path, detections, depth_scale, save_artifacts=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def run_worker(python_executable, worker_name, image_path, result_path,
               detections_path=None, depth_scale=DEFAULT_DEPTH_SCALE):
    command = [
        str(python_executable), str(Path(__file__).resolve()),
        "--worker", worker_name, "--image", str(image_path), "--result", str(result_path),
    ]
    if worker_name == "metric":
        if detections_path is None:
            raise ValueError("Metric3D worker requires the RF-DETR detections JSON.")
        command.extend(["--detections-file", str(detections_path), "--depth-scale", str(depth_scale)])
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"{worker_name} worker failed with exit code {completed.returncode}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def process_still_image(image_path, depth_scale=DEFAULT_DEPTH_SCALE,
                        run_lanes=True, output_path=None, announce=True):
    check_paths(require_lane=run_lanes)
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    tag = f"{image_path.stem}_{time.time_ns()}"
    rf_result_path = TMP_DIR / f"rfdetr_{tag}.json"
    lane_result_path = TMP_DIR / f"curvelanes_{tag}.json"
    metric_result_path = TMP_DIR / f"metric3d_{tag}.json"
    temp_paths = [rf_result_path, lane_result_path, metric_result_path]

    try:
        run_worker(RF_PYTHON, "rf", image_path, rf_result_path)
        detections = load_json(rf_result_path)

        lanes = []
        if run_lanes:
            run_worker(LANE_PYTHON, "lane", image_path, lane_result_path)
            lanes = load_json(lane_result_path).get("lanes", [])
        selected_lanes = select_two_lane_boundaries(lanes, image.shape)

        run_worker(
            METRIC_PYTHON, "metric", image_path, metric_result_path,
            detections_path=rf_result_path, depth_scale=depth_scale,
        )
        metric_result = load_json(metric_result_path)
        output_path = Path(output_path) if output_path else OUTPUT_DIR / f"visentra_{image_path.stem}_result.jpg"
        annotated = draw_results(
            image, detections, selected_lanes, output_path,
            distances=metric_result.get("distances", []),
            depth_scale=depth_scale,
            status_lines=["Still-image RF-DETR + Metric3D result"],
        )
        if announce:
            print(f"Objects detected: {len(detections.get('boxes', []))}; "
                  f"selected lane boundaries: {len(selected_lanes)}")
            skipped_non_vehicles = 0
            for item in metric_result.get("distances", []):
                if item.get("status") == "non_vehicle_skipped":
                    skipped_non_vehicles += 1
                    continue
                est = item.get("distance_m")
                name = item.get("class_name", "vehicle")
                conf = item.get("confidence")
                if est is None:
                    print(f"  Vehicle detection {item['detection_index']} ({name}): "
                          f"distance unavailable; {item.get('status')}")
                else:
                    conf_text = f", confidence={conf:.2f}" if conf is not None else ""
                    print(f"  Vehicle detection {item['detection_index']} ({name}{conf_text}): "
                          f"~{est:.2f} m [experimental]")
            if skipped_non_vehicles:
                print(f"Ignored {skipped_non_vehicles} non-vehicle detection(s) for distance estimation.")
            print(f"Saved annotated image: {output_path}")
            if metric_result.get("raw_depth_path"):
                print(f"Saved raw depth: {metric_result['raw_depth_path']}")
                print(f"Saved depth visualization: {metric_result['depth_visualization_path']}")
        return {
            "image_path": str(image_path),
            "detections": detections,
            "lanes": selected_lanes,
            "metric": metric_result,
            "output_path": str(output_path),
            "annotated_image": annotated,
        }
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def master(image_path, depth_scale=DEFAULT_DEPTH_SCALE, run_lanes=True):
    return process_still_image(image_path, depth_scale=depth_scale, run_lanes=run_lanes)


# ============================================================
# LOCAL TARGET SELECTION (NO IMAGE UPLOAD REQUIRED)
# ============================================================
def choose_reference_vehicle_index(label, image_path, result):
    """Show detections locally, then let the user choose the known-distance target."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for target selection: {image_path}")

    detections = result.get("detections", {})
    distance_items = result.get("metric", {}).get("distances", [])
    distance_by_index = {int(item.get("detection_index", -1)): item for item in distance_items}
    boxes = detections.get("boxes", []) or []
    names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []
    valid_vehicle_indices = []

    preview = image.copy()
    for index, box in enumerate(boxes):
        name = names[index] if index < len(names) else "object"
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        is_vehicle = is_vehicle_class(name)
        if is_vehicle:
            valid_vehicle_indices.append(index)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(x1, preview.shape[1] - 1))
        x2 = max(0, min(x2, preview.shape[1] - 1))
        y1 = max(0, min(y1, preview.shape[0] - 1))
        y2 = max(0, min(y2, preview.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 210, 0) if is_vehicle else (130, 130, 130)
        entry = distance_by_index.get(index, {})
        estimate = entry.get("distance_m")
        if is_vehicle and estimate is not None:
            label_text = f"#{index} {name} {confidence:.2f} ~{float(estimate):.1f}m"
        elif is_vehicle:
            label_text = f"#{index} {name} {confidence:.2f} distance unavailable"
        else:
            label_text = f"#{index} {name} {confidence:.2f} [not a vehicle target]"

        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        text_y = max(22, y1 - 8)
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )
        bg_x2 = min(preview.shape[1] - 1, x1 + text_w + 8)
        bg_y1 = max(0, text_y - text_h - 7)
        cv2.rectangle(preview, (x1, bg_y1), (bg_x2, text_y + 4), color, -1)
        cv2.putText(preview, label_text, (x1 + 3, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)

    header = f"{label}: identify the intended target vehicle; IDs match terminal prompt"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (30, 30, 30), -1)
    cv2.putText(preview, header[:110], (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)

    max_w, max_h = 1100, 720
    scale = min(max_w / preview.shape[1], max_h / preview.shape[0], 1.0)
    display = preview
    if scale < 1.0:
        display = cv2.resize(
            preview, (int(preview.shape[1] * scale), int(preview.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    window_name = f"VISENTRA target selection - {label}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display)
        print(f"\n[{label}] A local window shows numbered detections and distances.")
        print("Identify the SAME vehicle whose reference distance was measured.")
        print("Press any key while the image window is active; then enter its # ID in PowerShell.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        raise RuntimeError(
            "Could not open the local target-selection window. Run this command from an "
            "interactive Windows desktop session. No image upload is needed."
        ) from exc

    available = sorted(set(valid_vehicle_indices))
    if not available:
        print(f"No supported vehicle detection is available for {label}.")
        return None

    while True:
        raw = input(
            f"Enter target vehicle detection ID for {label} {available} "
            "(Enter = skip): "
        ).strip()
        if not raw:
            return None
        try:
            index = int(raw.lstrip("#"))
        except ValueError:
            print("Enter an integer detection ID, or press Enter to skip.")
            continue
        if index not in available:
            print(f"That ID is not a supported vehicle detection. Choose one of {available}.")
            continue
        return index


# ============================================================
# KNOWN-DISTANCE TEST MODE (RF-DETR FIRST, THEN METRIC3D)
# ============================================================
def test_known_distances(depth_scale=DEFAULT_DEPTH_SCALE, with_lanes=False,
                         select_targets=False):
    test_root = ROOT / r"MonoPilot\out\independent_tests"
    test_cases = [("5m", 5.0), ("10m", 10.0), ("15m", 15.0), ("20m", 20.0)]
    report_rows = []
    target_rows = []

    for label, reference_distance in test_cases:
        image_path = test_root / label / f"{label}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Known-distance image not found: {image_path}")
        print("\n" + "=" * 65)
        print(f"KNOWN-DISTANCE TEST: {label} (reference distance {reference_distance:.1f} m)")
        output_path = OUTPUT_DIR / f"visentra_{label}_rfdetr_metric3d.jpg"
        result = process_still_image(
            image_path,
            depth_scale=depth_scale,
            run_lanes=with_lanes,
            output_path=output_path,
        )
        distances = result["metric"].get("distances", [])
        vehicle_distances = [
            item for item in distances
            if item.get("status") != "non_vehicle_skipped"
        ]

        # Optional local-only target selection. The image stays on the user's PC.
        selected_index = None
        if select_targets:
            selected_index = choose_reference_vehicle_index(label, image_path, result)
            selected = next(
                (item for item in vehicle_distances
                 if int(item.get("detection_index", -1)) == selected_index),
                None,
            ) if selected_index is not None else None
            if selected is None:
                target_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "target_detection_index": "",
                    "class_name": "",
                    "confidence": "",
                    "raw_depth_median": "",
                    "estimated_distance_m": "",
                    "error_m": "",
                    "absolute_error_m": "",
                    "status": "target_skipped_or_unavailable",
                    "output_image": str(output_path),
                })
            else:
                estimated = selected.get("distance_m")
                error = (float(estimated) - reference_distance) if estimated is not None else None
                target_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "target_detection_index": selected_index,
                    "class_name": selected.get("class_name", "vehicle"),
                    "confidence": selected.get("confidence", ""),
                    "raw_depth_median": selected.get("raw_depth_median", ""),
                    "estimated_distance_m": estimated if estimated is not None else "",
                    "error_m": round(error, 3) if error is not None else "",
                    "absolute_error_m": round(abs(error), 3) if error is not None else "",
                    "status": selected.get("status", "unknown"),
                    "output_image": str(output_path),
                })
                if estimated is None:
                    print(f"Selected target #{selected_index}, but its distance is unavailable.")
                else:
                    print(
                        f"Selected reference target #{selected_index}: "
                        f"estimated={float(estimated):.2f} m, "
                        f"reference={reference_distance:.2f} m, "
                        f"error={float(estimated) - reference_distance:+.2f} m"
                    )

        if not vehicle_distances:
            print("No supported vehicle-class detections returned by RF-DETR for this image.")
            report_rows.append({
                "image": label,
                "reference_distance_m": reference_distance,
                "detection_index": "",
                "class_name": "",
                "confidence": "",
                "raw_depth_median": "",
                "estimated_distance_m": "",
                "valid_pixel_fraction": "",
                "capped_pixel_fraction": "",
                "status": "no_vehicle_detections",
                "output_image": str(output_path),
            })
        else:
            for item in vehicle_distances:
                report_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "detection_index": item.get("detection_index"),
                    "class_name": item.get("class_name"),
                    "confidence": item.get("confidence"),
                    "raw_depth_median": item.get("raw_depth_median"),
                    "estimated_distance_m": item.get("distance_m"),
                    "valid_pixel_fraction": item.get("valid_pixel_fraction"),
                    "capped_pixel_fraction": item.get("capped_pixel_fraction"),
                    "status": item.get("status"),
                    "output_image": str(output_path),
                })

    report_path = OUTPUT_DIR / "visentra_rfdetr_metric3d_known_distance_report.csv"
    if report_rows:
        report_fields = [
            "image", "reference_distance_m", "detection_index", "class_name",
            "confidence", "raw_depth_median", "estimated_distance_m",
            "valid_pixel_fraction", "capped_pixel_fraction", "status", "output_image",
        ]
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=report_fields)
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n========== FOUR-IMAGE REPORT ==========")
    for row in report_rows:
        estimated = row.get("estimated_distance_m")
        if row.get("status") == "no_vehicle_detections":
            print(f"{row['image']}: no supported vehicle detection")
        elif estimated in (None, ""):
            print(f"{row['image']}: vehicle det#{row.get('detection_index')}; "
                  f"distance unavailable ({row['status']})")
        else:
            print(f"{row['image']} reference={float(row['reference_distance_m']):.1f} m | "
                  f"vehicle det#{row['detection_index']} {row['class_name']} | "
                  f"Metric3D estimate ~{float(estimated):.2f} m (experimental)")
    print("Report CSV:", report_path)
    print("Annotated images:", OUTPUT_DIR)

    if select_targets:
        selected_report_path = OUTPUT_DIR / "visentra_rfdetr_metric3d_selected_target_validation.csv"
        selected_fields = [
            "image", "reference_distance_m", "target_detection_index", "class_name",
            "confidence", "raw_depth_median", "estimated_distance_m", "error_m",
            "absolute_error_m", "status", "output_image",
        ]
        with selected_report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected_fields)
            writer.writeheader()
            writer.writerows(target_rows)
        print("Local target-specific report CSV:", selected_report_path)
        print("\n========== SELECTED TARGET SUMMARY ==========")
        scored_rows = [row for row in target_rows if row.get("absolute_error_m") not in (None, "")]
        for row in target_rows:
            if row.get("absolute_error_m") in (None, ""):
                print(f"{row['image']}: {row['status']}")
            else:
                print(
                    f"{row['image']} target=#{row['target_detection_index']} "
                    f"reference={float(row['reference_distance_m']):.1f} m | "
                    f"estimate={float(row['estimated_distance_m']):.2f} m | "
                    f"error={float(row['error_m']):+.2f} m"
                )
        if scored_rows:
            mae = sum(float(row["absolute_error_m"]) for row in scored_rows) / len(scored_rows)
            print(f"Selected targets scored: {len(scored_rows)}; MAE on these same four calibration scenes: {mae:.2f} m")
        print("Reminder: this report uses the existing provisional scale and is not independent validation.")

    print("Note: only the manually selected target detection is compared to each reference distance.")


# ============================================================
# PERSISTENT LIVE-VIDEO WORKERS
# ============================================================
WORKER_READY_TIMEOUT_S = 600.0     # first start can be slow while models load on CPU
WORKER_REQUEST_TIMEOUT_S = 180.0
WORKER_STOP_GRACE_S = 6.0          # time a worker gets to exit by itself after stdin is closed
DETECTION_STALE_SECONDS = 5.0      # RF-DETR results older than this (since CAPTURE) are not drawn
LANE_STALE_SECONDS = 8.0           # same for lane results
DISTANCE_CARRY_SECONDS = 4.0       # per-track carry-over of a Metric3D value, flagged stale
LIVE_HISTORY_LIMIT = 16
CAPTURE_TS_KEEP_FRAMES = 1500      # capture timestamps remembered per frame ID
PERF_CSV_INTERVAL_S = 5.0
QUIT_KEYS = (ord("q"), ord("Q"), 27)
PERF_TOGGLE_KEYS = (ord("p"), ord("P"))


def configure_worker_threads(worker_name):
    """Apply the per-worker PyTorch thread count chosen by the parent (RF-DETR / CurveLaneS)."""
    try:
        count = int(os.environ.get("VISENTRA_TORCH_THREADS", "0") or 0)
    except ValueError:
        count = 0
    if count <= 0 or worker_name not in ("rf", "lane"):
        return
    try:
        import torch
        torch.set_num_threads(count)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialised; harmless
    except Exception:  # noqa: BLE001
        pass


def preload_stream_worker(worker_name):
    """Load the worker's model once at startup so the first live frame is not delayed."""
    configure_worker_threads(worker_name)
    if worker_name == "rf":
        get_rf_detector()
        if MOBILENETV3_ENABLED:
            # Load it now instead of stalling the 10th live request.
            try:
                load_mobilenetv3()
            except Exception as exc:  # noqa: BLE001 - optional model
                print(f"[MobileNetV3] Could not preload: {exc}", file=sys.stderr, flush=True)
    elif worker_name == "lane":
        cond_root = str(COND_LANE_ROOT)
        if cond_root not in sys.path:
            sys.path.insert(0, cond_root)
        from tools.condlanenet.post_process import CurvelanesPostProcessor  # noqa: F401
        get_lane_detector()
    elif worker_name == "metric":
        get_metric_estimator()
    else:
        raise ValueError(f"Unknown worker: {worker_name}")


def warmup_stream_worker(worker_name, depth_scale):
    """Run two throw-away inferences so one-time initialisation costs (allocator, kernel
    selection, lazy imports) are paid before the first real frame, not during it."""
    size_text = os.environ.get("VISENTRA_WARMUP_SIZE", "640x480")
    try:
        width, height = [int(v) for v in size_text.lower().split("x")]
    except ValueError:
        width, height = 640, 480
    rng = np.random.default_rng(12345)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    path = TMP_DIR / f"warmup_{worker_name}_{os.getpid()}.jpg"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write warm-up image: {path}")
    try:
        for _ in range(2):
            if worker_name == "rf":
                run_rfdetr(path)
            elif worker_name == "lane":
                run_curvelanes(path)
            elif worker_name == "metric":
                run_metric3d(path, EMPTY_DETECTIONS, depth_scale, save_artifacts=False)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


# ---------------- worker-side diagnostics ----------------
_WORKER_LOG_FH = None
_WORKER_CTX = {"name": "?", "frame": None, "stage": "startup"}


def wlog(message, console=False):
    """Write a diagnostic line (worker name, PID, frame, stage) to the worker's own log file.

    Goes to stderr only when console=True; stdout is never touched (JSON protocol only).
    """
    line = (f"{_timestamp()} [{_WORKER_CTX['name']} pid {os.getpid()}] "
            f"[frame {_WORKER_CTX['frame']}] [stage {_WORKER_CTX['stage']}] {message}")
    handle = _WORKER_LOG_FH
    if handle is not None:
        try:
            handle.write(line + "\n")
            handle.flush()
        except (OSError, ValueError):
            pass
    if console:
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass


def _rss_mb():
    """Resident memory of this process in MB, or None if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1048576.0
    except Exception:  # noqa: BLE001
        return None


def worker_stream(worker_name, depth_scale=DEFAULT_DEPTH_SCALE):
    global _WORKER_LOG_FH

    # Preserve stdout for JSON protocol messages.
    json_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    protocol_out = os.fdopen(
        json_fd, "w", encoding="utf-8", buffering=1, newline="\n"
    )

    log_path = os.environ.get("VISENTRA_WORKER_LOG") or str(
        WORKER_LOG_DIR / f"{SESSION_TAG}_{worker_name}.log"
    )

    try:
        _WORKER_LOG_FH = open(
            log_path, "a", encoding="utf-8", buffering=1
        )
        faulthandler.enable(file=_WORKER_LOG_FH, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        _WORKER_LOG_FH = None
        try:
            faulthandler.enable()
        except (OSError, RuntimeError, ValueError):
            pass

    _WORKER_CTX.update(
        name=worker_name,
        frame=None,
        stage="startup",
    )

    # The parent controls shutdown of persistent workers.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    def emit(message):
        protocol_out.write(json.dumps(message) + "\n")
        protocol_out.flush()

    exit_reason = "unknown"
    clean_exit = False

    wlog(
        f"worker starting (python {sys.version.split()[0]}, log {log_path})",
        console=True,
    )

    try:
        # Load each model once before accepting inference requests.
        try:
            _WORKER_CTX["stage"] = "preload"
            load_started = time.perf_counter()
            preload_stream_worker(worker_name)
            load_s = time.perf_counter() - load_started
        except Exception as exc:
            error_text = traceback.format_exc()
            sys.stderr.write(error_text)
            sys.stderr.flush()
            wlog("model preload failed:\n" + error_text)

            emit({
                "ready": False,
                "worker": worker_name,
                "error": repr(exc),
            })

            exit_reason = f"preload failed: {exc!r}"
            raise SystemExit(1)

        # Warm-up is timed separately from loading. A warm-up failure is logged but does not
        # prevent the worker from serving requests (real requests report their own errors).
        warmup_s = None
        try:
            _WORKER_CTX["stage"] = "warmup"
            warm_started = time.perf_counter()
            warmup_stream_worker(worker_name, depth_scale)
            warmup_s = time.perf_counter() - warm_started
        except Exception:
            error_text = traceback.format_exc()
            sys.stderr.write(error_text)
            sys.stderr.flush()
            wlog("warm-up failed (continuing):\n" + error_text)

        _WORKER_CTX["stage"] = "idle"
        emit({
            "ready": True,
            "worker": worker_name,
            "load_s": load_s,
            "warmup_s": warmup_s,
        })
        wlog(f"ready; load {load_s:.2f}s, warm-up "
             f"{'n/a' if warmup_s is None else f'{warmup_s:.2f}s'}; rss={_rss_mb()} MB")

        # Count RF-DETR requests, not camera frame IDs.
        rf_request_count = 0

        # There must be exactly ONE input loop.
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            frame_id = None

            try:
                _WORKER_CTX["stage"] = "read-request"
                request = json.loads(raw_line)
                frame_id = request.get("frame_id")
                _WORKER_CTX["frame"] = frame_id

                image_path = request.get("image_path")
                if not image_path:
                    raise ValueError("Worker request has no image_path.")

                started = time.perf_counter()
                wlog(
                    f"request received; image={image_path}; "
                    f"rss={_rss_mb()} MB"
                )

                if worker_name == "rf":
                    rf_request_count += 1

                    _WORKER_CTX["stage"] = "rfdetr-predict"
                    result = run_rfdetr(image_path)

                    # Classify crops on every tenth RF-DETR request.
                    if (
                        MOBILENETV3_ENABLED
                        and MOBILENETV3_EVERY_N_FRAMES > 0
                        and rf_request_count % MOBILENETV3_EVERY_N_FRAMES == 0
                    ):
                        _WORKER_CTX["stage"] = "mobilenetv3"
                        wlog(
                            f"running MobileNetV3 on RF request "
                            f"#{rf_request_count}"
                        )
                        result = attach_mobilenetv3_results(
                            image_path, result
                        )

                elif worker_name == "lane":
                    _WORKER_CTX["stage"] = "curvelanes"
                    result = {"lanes": run_curvelanes(image_path)}

                elif worker_name == "metric":
                    _WORKER_CTX["stage"] = "metric3d"
                    detections = request.get("detections")
                    if not isinstance(detections, dict):
                        raise ValueError(
                            "Metric3D request has no detections object."
                        )

                    result = run_metric3d(
                        image_path,
                        detections,
                        depth_scale,
                        save_artifacts=False,
                    )

                else:
                    raise ValueError(f"Unknown worker: {worker_name}")

                # Pure worker-side time (image decode + model + post-processing), measured
                # separately from the parent's round-trip time (JPEG write + pipe + this).
                worker_ms = (time.perf_counter() - started) * 1000.0
                _WORKER_CTX["stage"] = "emit-response"
                emit({
                    "ok": True,
                    "frame_id": frame_id,
                    "worker_ms": worker_ms,
                    "result": result,
                })

                _WORKER_CTX["stage"] = "idle"
                wlog(
                    f"done in {worker_ms / 1000.0:.2f}s; rss={_rss_mb()} MB"
                )

            except Exception as exc:
                # Recoverable request errors are returned to the parent.
                error_text = traceback.format_exc()
                sys.stderr.write(error_text)
                sys.stderr.flush()
                wlog("recoverable inference error:\n" + error_text)

                emit({
                    "ok": False,
                    "frame_id": frame_id,
                    "error": repr(exc),
                })
                _WORKER_CTX["stage"] = "idle"

            except BaseException as exc:
                # Fatal exceptions must remain visible in the worker log.
                error_text = traceback.format_exc()
                sys.stderr.write(error_text)
                sys.stderr.flush()
                wlog(
                    f"FATAL {type(exc).__name__} escaped inference:\n"
                    + error_text,
                    console=True,
                )

                try:
                    emit({
                        "ok": False,
                        "fatal": True,
                        "frame_id": frame_id,
                        "error": repr(exc),
                    })
                except Exception:
                    pass

                raise

        exit_reason = "stdin closed by the parent (clean shutdown)"
        clean_exit = True

    except BaseException as exc:
        if exit_reason == "unknown":
            exit_reason = f"{type(exc).__name__}: {exc!r}"
            wlog(
                "worker leaving with exception:\n" + traceback.format_exc(),
                console=True,
            )
        raise

    finally:
        wlog(
            f"worker exiting: {exit_reason}; rss={_rss_mb()} MB",
            console=not clean_exit,
        )

        try:
            sys.stderr.flush()
            protocol_out.flush()
            if _WORKER_LOG_FH is not None:
                _WORKER_LOG_FH.flush()
        except Exception:
            pass

    # Retain the existing Windows-specific clean-exit behavior.
    os._exit(0)


class StreamWorker:
    """Parent-side handle of one persistent inference subprocess.

    Requests are strictly serialized. Stdout lines that are not JSON are logged and ignored.
    A response is accepted only if its frame_id equals the request's frame_id; late answers
    to earlier (timed-out) requests are discarded.

    The stdout reader classifies the worker's end: if the process ends while VISENTRA is
    still running, that is recorded as a FIRST FAILURE with its PID, exit code and the tail
    of its log; if it ends after shutdown began, it is only a consequence of cleanup.
    """

    def __init__(self, name, depth_scale=DEFAULT_DEPTH_SCALE, command=None,
                 threads=None, warmup_size=None):
        self.name = name
        if command is None:
            python_executable = {
                "rf": RF_PYTHON, "lane": LANE_PYTHON, "metric": METRIC_PYTHON,
            }[name]
            command = [
                str(python_executable), str(Path(__file__).resolve()),
                "--worker-stream", name, "--depth-scale", str(depth_scale),
            ]
        self.log_path = WORKER_LOG_DIR / f"{SESSION_TAG}_{name}.log"
        thread_count = max(1, int((threads or DEFAULT_THREADS).get(name, 2)))
        self.thread_count = thread_count
        env = os.environ.copy()
        # Per-worker CPU thread pools (see --threads) so the three CPU models do not
        # oversubscribe the laptop.
        env["OMP_NUM_THREADS"] = str(thread_count)
        env["MKL_NUM_THREADS"] = str(thread_count)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["VISENTRA_TORCH_THREADS"] = str(thread_count)
        env["VISENTRA_ORT_THREADS"] = str(thread_count)
        env["PYTHONUNBUFFERED"] = "1"
        env["VISENTRA_WORKER_LOG"] = str(self.log_path)
        env["VISENTRA_MOBILENET"] = "1" if MOBILENETV3_ENABLED else "0"
        if warmup_size:
            env["VISENTRA_WARMUP_SIZE"] = str(warmup_size)
        self.process = subprocess.Popen(
            command, cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.pid = self.process.pid
        self.exit_code = None
        self.exit_class = None      # None (running) | "independent" | "after_shutdown"
        self.stop_requested = False
        self.stop_how = None
        self.load_s = None
        self.warmup_s = None
        self._responses = queue.Queue()
        self._ready = threading.Event()
        self._ready_ok = False
        self.ready_error = None
        self._request_lock = threading.Lock()
        parent_log(f"{name} worker started: PID {self.pid}; threads {thread_count}; "
                   f"log {self.log_path}")
        self._reader = threading.Thread(
            target=self._read_stdout, daemon=True, name=f"VISENTRA-{name}-stdout"
        )
        self._reader.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    log_warn(f"{self.name} worker wrote non-JSON text to stdout: {line[:200]}")
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    log_warn(f"{self.name} worker wrote an invalid JSON line: {line[:200]}")
                    continue
                if "ready" in message:
                    self._ready_ok = bool(message["ready"])
                    self.ready_error = None if self._ready_ok else message.get("error", "unknown error")
                    if self._ready_ok:
                        self.load_s = message.get("load_s")
                        self.warmup_s = message.get("warmup_s")
                        log_info(
                            f"{self.name} worker ready (PID {self.pid}, {self.thread_count} threads): "
                            f"model load {_fmt(self.load_s, '.1f', ' s')}, "
                            f"warm-up {_fmt(self.warmup_s, '.1f', ' s')}"
                        )
                    self._ready.set()
                    continue
                self._responses.put(message)
        except (OSError, ValueError):
            pass
        finally:
            # Decide NOW whether this end is a failure or a consequence of shutdown; the wait
            # below can take time and must not blur that decision.
            shutdown_already_begun = SHUTDOWN.is_set() or self.stop_requested
            try:
                code = self.process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                code = None
            self.exit_code = code
            if shutdown_already_begun:
                self.exit_class = "after_shutdown"
                parent_log(
                    f"{self.name} worker (PID {self.pid}) ended with code {code} after shutdown "
                    f"began (reason: {SHUTDOWN.reason}); not an independent failure. "
                    "On Windows a worker stopped with terminate() reports code 1."
                )
            else:
                self.exit_class = "independent"
                tail = tail_text(self.log_path, 25)
                detail = (
                    f"{self.name} worker (PID {self.pid}) exited ON ITS OWN with exit code {code} "
                    "while VISENTRA was still running and no shutdown had been requested. "
                    f"Worker log: {self.log_path}"
                )
                if tail:
                    detail += f"\n--- last lines of the {self.name} worker log ---\n{tail}\n--- end ---"
                else:
                    detail += (" (the log is empty: the process ended before it could write; "
                               "a native crash would normally appear there via faulthandler)")
                SHUTDOWN.failure(detail)
            if not self._ready.is_set():
                self.ready_error = f"worker exited with code {code} before it was ready"
                self._ready.set()
            self._responses.put(None)  # wake any waiting request

    def is_ready(self):
        return self._ready.is_set() and self._ready_ok

    def has_failed(self):
        return self._ready.is_set() and not self._ready_ok

    def alive(self):
        return self.process.poll() is None

    def request(self, payload, timeout=WORKER_REQUEST_TIMEOUT_S):
        expected = payload["frame_id"]
        with self._request_lock:
            if not self.alive():
                if SHUTDOWN.is_set() or self.stop_requested:
                    raise WorkerShutdownError(f"{self.name} worker stopped during shutdown")
                raise WorkerExitedError(
                    f"{self.name} worker (PID {self.pid}) is not running "
                    f"(exit code {self.process.poll()}; log {self.log_path})")
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                if SHUTDOWN.is_set() or self.stop_requested:
                    raise WorkerShutdownError(f"{self.name} worker stopped during shutdown") from exc
                raise WorkerExitedError(
                    f"could not send a request to the {self.name} worker "
                    f"(PID {self.pid}): {exc}") from exc
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.name} worker gave no answer for frame {expected} within {timeout:.0f}s")
                try:
                    message = self._responses.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if message is None:
                    self._responses.put(None)
                    if self.exit_class == "after_shutdown" or SHUTDOWN.is_set():
                        raise WorkerShutdownError(
                            f"{self.name} worker stopped during shutdown "
                            f"(exit code {self.exit_code})")
                    raise WorkerExitedError(
                        f"{self.name} worker (PID {self.pid}) closed its output stream while "
                        f"processing frame {expected} (exit code: {self.exit_code}); "
                        f"log: {self.log_path}"
                    )
                if message.get("fatal"):
                    # The worker reported that a non-Exception (e.g. SystemExit) escaped inference
                    # and is about to exit. Give it a moment so the exit code is known.
                    try:
                        self.process.wait(timeout=5)
                    except Exception:  # noqa: BLE001
                        pass
                    raise WorkerExitedError(
                        f"{self.name} worker (PID {self.pid}) reported a fatal error on frame "
                        f"{message.get('frame_id')}: {message.get('error')} "
                        f"(exit code {self.process.poll()}; log {self.log_path})"
                    )
                if message.get("frame_id") != expected:
                    log_warn(
                        f"{self.name}: discarding a late response for frame "
                        f"{message.get('frame_id')} while waiting for frame {expected}"
                    )
                    continue
                return message

    def request_stop(self):
        """Ask the worker to finish: closing stdin ends its read loop (clean exit code 0)."""
        self.stop_requested = True
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self, grace=WORKER_STOP_GRACE_S):
        """Graceful stop first; terminate/kill only if the worker does not exit in time."""
        self.request_stop()
        try:
            code = self.process.wait(timeout=grace)
            self.stop_how = f"exited by itself after stdin closed (code {code})"
        except subprocess.TimeoutExpired:
            self.stop_how = _stop_process(self.process)
        except Exception as exc:  # noqa: BLE001
            self.stop_how = f"wait failed: {exc!r}"
        parent_log(f"{self.name} worker (PID {self.pid}) stop result: {self.stop_how}")
        return self.stop_how


def _put_latest(work_queue, item):
    """Bounded 'latest wins' queue. Returns True if an older queued item was discarded."""
    replaced = False
    if work_queue.full():
        try:
            work_queue.get_nowait()
            replaced = True
        except queue.Empty:
            pass
    try:
        work_queue.put_nowait(item)
    except queue.Full:
        pass
    return replaced


def _stop_process(process):
    """Terminate with a timeout, then force-kill. Only the given process is touched.

    Returns a description of what happened. NOTE: on Windows terminate() makes the child exit
    with code 1, so that code means "stopped by VISENTRA", not "crashed".
    """
    if process is None:
        return "no process"
    try:
        if process.poll() is not None:
            return f"already exited (code {process.returncode})"
        process.terminate()
        process.wait(timeout=5)
        return (f"terminated by VISENTRA (code {process.returncode}; "
                "Windows reports 1 for a terminated process)")
    except Exception:  # noqa: BLE001
        try:
            process.kill()
            process.wait(timeout=3)
            return f"force-killed by VISENTRA (code {process.returncode})"
        except Exception:  # noqa: BLE001
            return "could not be stopped"


def _trim_history(history, limit=LIVE_HISTORY_LIMIT):
    while len(history) > limit:
        history.pop(next(iter(history)))


def update_distance_cache(cache, distances, tracked_detections, completed_at):
    """Remember each tracked vehicle's latest Metric3D value, keyed by ByteTrack ID."""
    track_ids = tracked_detections.get("track_ids", []) or []
    for entry in distances or []:
        index = int(entry.get("detection_index", -1))
        value = entry.get("distance_m")
        if value is None or index < 0 or index >= len(track_ids):
            continue
        track_id = track_ids[index]
        if track_id is None:
            continue
        cache[track_id] = {"distance_m": float(value), "completed_at": completed_at}


def assemble_live_distances(detections, frame_id, exact, cache, now):
    """Pair Metric3D distances with the detections being drawn.

    * exact: the Metric3D result computed on the SAME frame (strict match), or None.
    * otherwise a recent value for the same ByteTrack ID is carried over and flagged stale.
    * nothing is ever paired by box position across different frames.
    """
    names = detections.get("class_names", []) or []
    track_ids = detections.get("track_ids", []) or []
    by_index = {}
    if exact is not None:
        by_index = {int(item.get("detection_index", -1)): item for item in exact}
    entries = []
    for index in range(len(detections.get("boxes", []) or [])):
        if index >= len(names) or not is_vehicle_class(names[index]):
            continue
        if exact is not None and index in by_index:
            entries.append(by_index[index])
            continue
        track_id = track_ids[index] if index < len(track_ids) else None
        cached = cache.get(track_id) if track_id is not None else None
        if cached and now - cached["completed_at"] <= DISTANCE_CARRY_SECONDS:
            entries.append({
                "detection_index": index, "distance_m": cached["distance_m"],
                "stale": True, "status": "carried_over_by_track_id",
            })
        else:
            entries.append({"detection_index": index, "distance_m": None, "status": "pending"})
    return entries


EMPTY_DETECTIONS = {"boxes": [], "class_ids": [], "class_names": [], "confidences": []}

PERF_CSV_FIELDS = [
    "elapsed_s", "capture_fps", "display_fps",
    "rf_fps", "rf_worker_ms", "rf_roundtrip_ms",
    "lane_fps", "lane_worker_ms", "lane_roundtrip_ms",
    "metric_fps", "metric_worker_ms", "metric_roundtrip_ms",
    "bytetrack_ms", "deepsort_ms", "e2e_capture_to_display_ms",
    "captured_frames", "dropped_before_rf", "dropped_before_lane",
    "dropped_before_metric", "dropped_before_track", "video_frames_skipped",
    "detection_age_s", "lane_age_s", "depth_age_s", "cpu_percent", "ram_mb",
]


def master_live(camera_index=0, width=640, height=480, depth_scale=DEFAULT_DEPTH_SCALE,
                with_lanes=False, serial=None, camera_fps=15, video_path=None,
                threads=None, lane_every=1, metric_every=1, enable_deepsort=False,
                detection_max_age=DETECTION_STALE_SECONDS, lane_max_age=LANE_STALE_SECONDS):
    """Responsive live or recorded-video preview. One input, one worker per model, bounded queues.

    lane_every / metric_every: send every N-th completed RF-DETR frame to CurveLaneS / Metric3D
    (1 = every one the worker can take; the models are never disabled by these settings).
    """
    global _PARENT_LOG_ENABLED
    _PARENT_LOG_ENABLED = True
    SHUTDOWN.reset()
    lane_every = max(1, int(lane_every))
    metric_every = max(1, int(metric_every))
    parent_log(f"live session start; session tag {SESSION_TAG}; lanes={with_lanes}; "
               f"{width}x{height}@{camera_fps}; MOBILENETV3_ENABLED={MOBILENETV3_ENABLED}; "
               f"DEEPSORT={enable_deepsort}; threads={threads or DEFAULT_THREADS}; "
               f"lane_every={lane_every}; metric_every={metric_every}")
    log_info(f"Diagnostic logs for this session: {WORKER_LOG_DIR} (files start with {SESSION_TAG})")
    log_info(
        f"Configuration: threads {threads or DEFAULT_THREADS}; CurveLaneS every {lane_every} RF frame(s); "
        f"Metric3D every {metric_every} RF frame(s); DeepSORT {'ON' if enable_deepsort else 'off'}; "
        f"MobileNetV3 {'ON' if MOBILENETV3_ENABLED else 'off'}."
    )

    check_paths(require_lane=with_lanes, require_camera=(video_path is None))
    cv2.setNumThreads(1)
    if video_path is None and camera_index not in (0, None):
        log_warn("--camera is ignored: the direct Vivo camera stream is used instead of a webcam index.")

    perf = PerfStats()
    monitor = ResourceMonitor()
    if not monitor.available:
        log_warn("psutil is not installed in this environment: CPU/RAM figures will show n/a "
                 "(install with: pip install psutil).")

    worker_names = ["rf"] + (["lane"] if with_lanes else []) + ["metric"]
    queues = {name: queue.Queue(maxsize=1) for name in worker_names + ["track"]}
    state_lock = threading.Lock()
    stop_event = threading.Event()
    fatal = {"message": None}   # set when a worker failure forces the live session to stop
    state = {
        "rf": {"frame_id": None, "completed_at": None, "fps": 0.0, "error": None},
        "lane": {"lanes": [], "frame_id": None, "completed_at": None, "fps": 0.0, "error": None},
        "metric": {"frame_id": None, "completed_at": None, "fps": 0.0, "error": None},
        "track": {"frame_id": None, "detections": None, "completed_at": None, "error": None},
    }
    metric_history = {}    # frame_id -> (completed_at, distances)
    tracked_history = {}   # frame_id -> detections with ByteTrack/DeepSORT IDs
    distance_cache = {}    # ByteTrack ID -> latest Metric3D value
    cached_frames = set()
    capture_ts = {}        # frame_id -> perf_counter() when that frame was captured

    camera_session = None
    cap = None
    workers = {}
    threads_list = []
    latest_annotated = None
    bytetrack = None
    deepsort_tracker = None
    csv_handle = None
    csv_writer = None
    warmup_size = f"{width}x{height}"
    source_fps = None
    window_name = (
        f"VISENTRA - {Path(video_path).name}" if video_path is not None
        else "VISENTRA - Live Rear Camera"
    )

    def update_rate(name, completed):
        state[name]["completed_at"] = completed
        perf.tick(name, completed)
        state[name]["fps"] = perf.fps(name) or 0.0

    def source_time(frame_id, fallback):
        """Capture time of the frame a result was computed from (fallback: completion time)."""
        captured = capture_ts.get(frame_id) if frame_id is not None else None
        return captured if captured is not None else fallback

    def inference_loop(name):
        worker = workers[name]
        completed_rf = 0
        while not stop_event.is_set():
            try:
                item = queues[name].get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, frame_id, detections = item
            frame_path = TMP_DIR / f"live_{name}_{frame_id}_{time.time_ns()}.jpg"
            try:
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write temporary frame: {frame_path}")
                payload = {"frame_id": frame_id, "image_path": str(frame_path)}
                if name == "metric":
                    payload["detections"] = detections
                t_send = time.perf_counter()
                response = worker.request(payload, WORKER_REQUEST_TIMEOUT_S)
                t_done = time.perf_counter()
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "unknown worker error"))
                result = response["result"]
                perf.add(f"{name}_roundtrip", (t_done - t_send) * 1000.0)
                if response.get("worker_ms") is not None:
                    perf.add(f"{name}_worker", float(response["worker_ms"]))
                with state_lock:
                    update_rate(name, t_done)
                    state[name]["frame_id"] = frame_id
                    state[name]["error"] = None
                    if name == "lane":
                        state["lane"]["lanes"] = result.get("lanes", [])
                    elif name == "metric":
                        metric_history[frame_id] = (t_done, result.get("distances", []))
                        _trim_history(metric_history)
                if name == "rf":
                    completed_rf += 1
                    # Downstream stages receive exactly this frame and its detections.
                    if _put_latest(queues["track"], (frame, frame_id, result)):
                        perf.count("dropped_track")
                    if completed_rf % metric_every == 0:
                        if _put_latest(queues["metric"], (frame, frame_id, result)):
                            perf.count("dropped_metric")
                    if with_lanes and completed_rf % lane_every == 0:
                        if _put_latest(queues["lane"], (frame, frame_id, None)):
                            perf.count("dropped_lane")
            except WorkerShutdownError as exc:
                # The worker ended because we were already shutting down: not a failure.
                parent_log(f"{name} inference stopped by shutdown: {exc}")
                break
            except Exception as exc:  # noqa: BLE001
                if SHUTDOWN.is_set():
                    # Anything raised after shutdown began is a consequence, never the cause.
                    parent_log(f"{name} error after shutdown began (ignored): {exc!r}")
                    break
                with state_lock:
                    state[name]["error"] = str(exc)
                log_warn_once(f"{name}:{exc}", f"{name} inference error: {exc}")
                if isinstance(exc, WorkerExitedError) or not worker.alive():
                    SHUTDOWN.failure(f"{name} inference thread: {exc}")
                    if name == "rf":
                        # Nothing useful can continue without detections.
                        fatal["message"] = f"RF-DETR worker stopped: {exc}"
                        SHUTDOWN.begin(f"RF-DETR worker failure: {exc}")
                        stop_event.set()
                    else:
                        log_error(f"{name} worker is no longer running: {exc}")
                    break
            finally:
                try:
                    frame_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    def track_loop():
        while not stop_event.is_set():
            try:
                item = queues["track"].get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, frame_id, detections = item
            try:
                t0 = time.perf_counter()
                tracked = update_vehicle_tracks(detections, bytetrack)       # ByteTrack
                t1 = time.perf_counter()
                perf.add("bytetrack", (t1 - t0) * 1000.0)
                if deepsort_tracker is not None:                             # optional DeepSORT
                    tracked = update_deepsort_tracks(frame, tracked, deepsort_tracker)
                    perf.add("deepsort", (time.perf_counter() - t1) * 1000.0)
                with state_lock:
                    state["track"].update(
                        frame_id=frame_id, detections=tracked,
                        completed_at=time.perf_counter(), error=None,
                    )
                    tracked_history[frame_id] = tracked
                    _trim_history(tracked_history)
            except Exception as exc:  # noqa: BLE001
                with state_lock:
                    state["track"]["error"] = str(exc)
                parent_log("tracking error traceback:\n" + traceback.format_exc())
                log_warn_once(f"track:{exc}", f"tracking error: {exc}")

    def wait_for_workers():
        """Keep the live preview moving while the models load. Returns False if the user quits."""
        deadline = time.monotonic() + WORKER_READY_TIMEOUT_S
        while True:
            for name in worker_names:
                if workers[name].has_failed():
                    raise RuntimeError(
                        f"{name} worker failed to start: {workers[name].ready_error}"
                    )
            pending = [name for name in worker_names if not workers[name].is_ready()]
            if not pending:
                return True
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for worker(s): {pending}")
            if video_path is not None:
                # Do not consume the recording while CPU models are loading.
                view = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(view, "VISENTRA: loading models",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(view, f"Pending: {', '.join(pending)}",
                            (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, view)
                if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    return False
                time.sleep(0.01)
                continue
            ok, frame = cap.read(timeout=1.0)
            if not ok:
                if cap.failed:
                    stage, detail = cap.diagnose()
                    raise VivoCameraError(stage, detail)
                if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    return False
                continue
            view = frame.copy()
            cv2.putText(view, f"VISENTRA: loading models ({', '.join(pending)} pending)...",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, view)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                return False

    def write_perf_row(elapsed, det_age, lane_age_value, depth_age):
        nonlocal csv_handle, csv_writer
        try:
            if csv_writer is None:
                csv_handle = open(PERF_CSV_PATH, "w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(csv_handle, fieldnames=PERF_CSV_FIELDS)
                csv_writer.writeheader()

            def r(value, digits=2):
                return "" if value is None else round(float(value), digits)

            csv_writer.writerow({
                "elapsed_s": r(elapsed, 1),
                "capture_fps": r(perf.fps("capture")), "display_fps": r(perf.fps("display")),
                "rf_fps": r(perf.fps("rf")), "rf_worker_ms": r(perf.ms("rf_worker"), 1),
                "rf_roundtrip_ms": r(perf.ms("rf_roundtrip"), 1),
                "lane_fps": r(perf.fps("lane")), "lane_worker_ms": r(perf.ms("lane_worker"), 1),
                "lane_roundtrip_ms": r(perf.ms("lane_roundtrip"), 1),
                "metric_fps": r(perf.fps("metric")), "metric_worker_ms": r(perf.ms("metric_worker"), 1),
                "metric_roundtrip_ms": r(perf.ms("metric_roundtrip"), 1),
                "bytetrack_ms": r(perf.ms("bytetrack")), "deepsort_ms": r(perf.ms("deepsort")),
                "e2e_capture_to_display_ms": r(perf.ms("e2e"), 1),
                "captured_frames": perf.counter("captured"),
                "dropped_before_rf": perf.counter("dropped_rf"),
                "dropped_before_lane": perf.counter("dropped_lane"),
                "dropped_before_metric": perf.counter("dropped_metric"),
                "dropped_before_track": perf.counter("dropped_track"),
                "video_frames_skipped": perf.counter("video_skipped"),
                "detection_age_s": r(det_age), "lane_age_s": r(lane_age_value),
                "depth_age_s": r(depth_age),
                "cpu_percent": r(monitor.cpu, 1), "ram_mb": r(monitor.ram_mb, 0),
            })
            csv_handle.flush()
        except OSError:
            pass

    def log_perf_summary():
        elapsed = max(1e-6, time.perf_counter() - perf.t0)
        log_info(f"Measured session summary ({elapsed:.1f} s; averages over the whole session):")
        log_info(f"  capture {perf.total('capture') / elapsed:.2f} FPS | display "
                 f"{perf.total('display') / elapsed:.2f} FPS | frames captured "
                 f"{perf.counter('captured')} | not sent to RF-DETR (queue replaced) "
                 f"{perf.counter('dropped_rf')} | video frames skipped to keep real time "
                 f"{perf.counter('video_skipped')}")
        for label, key in (("RF-DETR", "rf"), ("CurveLaneS", "lane"), ("Metric3D", "metric")):
            done = perf.total(key)
            if key == "lane" and not with_lanes:
                continue
            log_info(
                f"  {label}: {done} results, {done / elapsed:.2f} FPS | worker latency mean "
                f"{_fmt(perf.ms_all(key + '_worker'), '.0f', ' ms')} | round-trip mean "
                f"{_fmt(perf.ms_all(key + '_roundtrip'), '.0f', ' ms')}"
            )
        log_info(f"  ByteTrack mean {_fmt(perf.ms_all('bytetrack'), '.1f', ' ms')} | DeepSORT mean "
                 f"{_fmt(perf.ms_all('deepsort'), '.1f', ' ms')} | capture->display mean "
                 f"{_fmt(perf.ms_all('e2e'), '.0f', ' ms')} | results discarded by newer ones: "
                 f"track {perf.counter('dropped_track')}, Metric3D {perf.counter('dropped_metric')}, "
                 f"CurveLaneS {perf.counter('dropped_lane')}")
        log_info(f"  Per-interval measurements are in: {PERF_CSV_PATH}")

    try:
        if video_path is not None:
            video_path = Path(video_path).resolve()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open recorded video: {video_path}")
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if not source_fps or not math.isfinite(source_fps) or source_fps < 1.0 or source_fps > 240.0:
                log_warn(f"The video reports an unusable frame rate ({source_fps}); assuming 30 FPS.")
                source_fps = 30.0
            video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if video_w > 0 and video_h > 0:
                warmup_size = f"{video_w}x{video_h}"
            log_info(
                f"Opened recorded video: {video_path} "
                f"({source_fps:.2f} FPS, {source_frames} frames, {warmup_size}). "
                "Playback runs at the original frame rate; frames are skipped, never slowed, "
                "when the display cannot keep up."
            )
        else:
            log_info("Starting Vivo rear camera automatically...")
            camera_session, cap = start_vivo_camera(width, height, camera_fps, serial)

        log_info("Starting persistent RF-DETR, "
                 + ("CurveLaneS, " if with_lanes else "")
                 + "and Metric3D workers (models load once, then warm up)...")
        for name in worker_names:
            workers[name] = StreamWorker(name, depth_scale, threads=threads, warmup_size=warmup_size)

        import supervision as sv

        # ByteTrack is always enabled.
        bytetrack = sv.ByteTrack()
        if enable_deepsort:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            # Optional independent DeepSORT tracker for comparison (costly: CNN embedder on CPU).
            deepsort_tracker = DeepSort(
                max_age=15, n_init=2, embedder="mobilenet", embedder_gpu=False, half=False,
            )

        if not wait_for_workers():
            SHUTDOWN.begin("user pressed Q/ESC while the models were loading")
            log_info("Stopped by the user while the models were loading.")
            return

        threads_list = [
            threading.Thread(target=inference_loop, args=(name,), daemon=True,
                             name=f"VISENTRA-{name}")
            for name in worker_names
        ]
        threads_list.append(threading.Thread(target=track_loop, daemon=True, name="VISENTRA-Tracking"))
        for thread in threads_list:
            thread.start()

        frame_number = 0
        video_t0 = None
        video_next_index = 0
        last_displayed_fid = None
        last_csv_time = time.perf_counter()
        show_perf = True
        perf.t0 = time.perf_counter()
        if video_path is not None:
            log_info(f"Recorded-video playback started ({'lanes on' if with_lanes else 'lanes off'}).")
        else:
            log_info(f"Live camera started ({'lanes on' if with_lanes else 'lanes off'}; "
                     f"requested {width}x{height} at {camera_fps} FPS).")
        log_info("RF-DETR runs first; Metric3D uses the same frame + its detection boxes.")
        log_info("Press Q or ESC to quit, P to hide/show the performance overlay.")

        while not stop_event.is_set():
            if video_path is not None:
                # Real-time pacing from the source frame rate: wait if ahead, skip if behind.
                if video_t0 is None:
                    video_t0 = time.perf_counter()
                elapsed_play = time.perf_counter() - video_t0
                due_time = video_next_index / source_fps
                if elapsed_play < due_time:
                    time.sleep(due_time - elapsed_play)
                else:
                    behind = int((elapsed_play - due_time) * source_fps)
                    if behind >= 1:
                        skipped = 0
                        for _ in range(behind):
                            if not cap.grab():
                                break
                            skipped += 1
                        video_next_index += skipped
                        perf.count("video_skipped", skipped)
                ok, frame = cap.read()
                if not ok or frame is None:
                    log_info(f"Reached end of recorded video: {Path(video_path).name}")
                    break
                video_next_index += 1
                frame_number = video_next_index          # source frame index (1-based)
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    stage, detail = cap.diagnose()
                    raise VivoCameraError(stage, detail)
                frame_number += 1

            capture_time = time.perf_counter()
            capture_ts[frame_number] = capture_time
            perf.tick("capture", capture_time)
            perf.count("captured")
            if frame_number % 300 == 0:
                cutoff = frame_number - CAPTURE_TS_KEEP_FRAMES
                for old_id in [k for k in capture_ts if k < cutoff]:
                    capture_ts.pop(old_id, None)

            if _put_latest(queues["rf"], (frame, frame_number, None)):
                perf.count("dropped_rf")

            now = time.perf_counter()
            display_fps = perf.fps("display")
            monitor.sample(now)

            with state_lock:
                rf_state = dict(state["rf"])
                lane_state = dict(state["lane"])
                metric_state = dict(state["metric"])
                track_state = dict(state["track"])
                for fid, (done_at, dists) in list(metric_history.items()):
                    if fid not in cached_frames and fid in tracked_history:
                        update_distance_cache(distance_cache, dists, tracked_history[fid], done_at)
                        cached_frames.add(fid)
                cached_frames.intersection_update(metric_history.keys())
                tracked = track_state["detections"]
                tracked_fid = track_state["frame_id"]
                exact = metric_history.get(tracked_fid)
                exact_distances = exact[1] if exact is not None else None
                lanes_now = list(lane_state["lanes"])

            status_lines = []
            detections_for_draw = EMPTY_DETECTIONS
            distances = None
            detection_age = None
            tracked_fresh = False
            if tracked is not None and track_state["completed_at"] is not None:
                # Age is measured from the CAPTURE of the frame the detections came from.
                detection_age = now - source_time(tracked_fid, track_state["completed_at"])
            if tracked is not None and detection_age is not None and detection_age <= detection_max_age:
                tracked_fresh = True
                detections_for_draw = tracked
                distances = assemble_live_distances(
                    tracked, tracked_fid, exact_distances, distance_cache, now)
                if exact_distances is not None:
                    depth_status = "Metric3D distance matched to this detection frame"
                else:
                    depth_status = "Metric3D distance pending for current detection frame (stale values marked *)"
                status_lines.append(
                    f"Detections: source frame {tracked_fid}, captured {detection_age:.1f}s ago "
                    f"(showing frame {frame_number})")
            elif tracked is not None:
                depth_status = "Metric3D: waiting (detections stale)"
                status_lines.append(
                    f"RF-DETR result from frame {tracked_fid} is {detection_age:.1f}s old "
                    f"(limit {detection_max_age:.1f}s); not drawn")
            else:
                depth_status = "Metric3D: waiting for the first RF-DETR result"
                status_lines.append("RF-DETR: waiting for the first result...")
            if metric_state["error"]:
                depth_status = "Metric3D worker error; check console"

            lane_age = None
            lane_fid = lane_state["frame_id"]
            if lane_state["completed_at"] is not None:
                lane_age = now - source_time(lane_fid, lane_state["completed_at"])
            if with_lanes:
                if lane_age is None:
                    lanes_now = []
                    lane_text = "CurveLaneS: pending"
                elif lane_age > lane_max_age:
                    lanes_now = []
                    lane_text = f"CurveLaneS: frame {lane_fid} is {lane_age:.1f}s old; not drawn"
                else:
                    lane_text = (f"CurveLaneS: source frame {lane_fid}, captured {lane_age:.1f}s ago "
                                 f"(drawn on frame {frame_number})")
            else:
                lanes_now = []
                lane_text = "CurveLaneS: off"
            status_lines.insert(0, lane_text)
            status_lines.append(depth_status)
            if rf_state["error"]:
                status_lines.append("RF-DETR worker error; check console")
            if with_lanes and lane_state["error"]:
                status_lines.append("CurveLaneS worker error; check console")
            if track_state["error"]:
                status_lines.append("Tracking error; check console")
            for stopped in worker_names:
                if stopped in workers and not workers[stopped].alive():
                    status_lines.append(f"{stopped} worker has stopped; check console")

            depth_age = None
            if metric_state["completed_at"] is not None:
                depth_age = now - source_time(metric_state["frame_id"], metric_state["completed_at"])

            if show_perf:
                status_lines.append(
                    f"Capture {_fmt(perf.fps('capture'), '.1f')} FPS | display {_fmt(display_fps, '.1f')} FPS | "
                    f"frames not sent to RF-DETR {perf.counter('dropped_rf')}/{perf.counter('captured')}"
                    + (f" | skipped {perf.counter('video_skipped')}" if video_path is not None else ""))
                status_lines.append(
                    f"RF-DETR {_fmt(perf.fps('rf'), '.2f')} FPS {_fmt(perf.ms('rf_worker'), '.0f')} ms "
                    f"(round trip {_fmt(perf.ms('rf_roundtrip'), '.0f')}) | ByteTrack "
                    f"{_fmt(perf.ms('bytetrack'), '.1f')} ms"
                    + (f" | DeepSORT {_fmt(perf.ms('deepsort'), '.0f')} ms" if enable_deepsort else ""))
                status_lines.append(
                    ("CurveLaneS "
                     f"{_fmt(perf.fps('lane'), '.2f')} FPS {_fmt(perf.ms('lane_worker'), '.0f')} ms | "
                     if with_lanes else "")
                    + f"Metric3D {_fmt(perf.fps('metric'), '.2f')} FPS {_fmt(perf.ms('metric_worker'), '.0f')} ms")
                status_lines.append(
                    f"Result age: det {_fmt(detection_age, '.1f')}s | lane {_fmt(lane_age, '.1f')}s | "
                    f"depth {_fmt(depth_age, '.1f')}s | capture->display {_fmt(perf.ms('e2e'), '.0f')} ms")
                status_lines.append(
                    f"CPU {_fmt(monitor.cpu, '.0f', '%')} | RAM (all processes) {_fmt(monitor.ram_mb, '.0f', ' MB')}")

            latest_annotated = draw_results(
                frame, detections_for_draw, lanes_now,
                distances=distances, depth_scale=depth_scale,
                frame_number=frame_number, fps=display_fps,
                status_lines=status_lines, lanes_enabled=with_lanes,
            )
            cv2.imshow(window_name, latest_annotated)
            shown_at = time.perf_counter()
            perf.tick("display", shown_at)
            if tracked_fresh and tracked_fid != last_displayed_fid:
                # First time this detection result reaches the screen: capture -> display delay.
                source_t = capture_ts.get(tracked_fid)
                if source_t is not None:
                    perf.add("e2e", (shown_at - source_t) * 1000.0)
                last_displayed_fid = tracked_fid

            if shown_at - last_csv_time >= PERF_CSV_INTERVAL_S:
                last_csv_time = shown_at
                write_perf_row(shown_at - perf.t0, detection_age, lane_age, depth_age)

            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                SHUTDOWN.begin("user pressed Q/ESC")
                break
            if key in PERF_TOGGLE_KEYS:
                show_perf = not show_perf

        if fatal["message"]:
            first = SHUTDOWN.first_failure
            message = fatal["message"]
            if first and first not in message:
                message += f"\nOriginal failure: {first}"
            raise RuntimeError(message)
    except BaseException as exc:  # noqa: BLE001 - record the ORIGINAL reason, then re-raise
        if SHUTDOWN.begin(f"{type(exc).__name__} in the live session: {exc}"):
            parent_log("live session exception:\n" + traceback.format_exc())
        raise
    finally:
        SHUTDOWN.begin("live session ended")   # no-op if a reason was already recorded
        stop_event.set()
        for work_queue in queues.values():
            _put_latest(work_queue, None)
        # Ask every worker to finish first (stdin EOF = clean exit) so they stop in parallel.
        for worker in workers.values():
            try:
                worker.request_stop()
            except Exception as exc:  # noqa: BLE001
                log_warn(f"{worker.name} worker stop request failed: {exc}")
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:  # noqa: BLE001
                log_warn(f"camera stream release failed: {exc}")
        if camera_session is not None:
            try:
                camera_session.close()
            except Exception as exc:  # noqa: BLE001
                log_warn(f"camera session cleanup failed: {exc}")
        for worker in workers.values():
            try:
                worker.close()
            except Exception as exc:  # noqa: BLE001
                log_warn(f"{worker.name} worker cleanup failed: {exc}")
        for thread in threads_list:
            thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if latest_annotated is not None:
            try:
                output_path = OUTPUT_DIR / "visentra_live_result.jpg"
                cv2.imwrite(str(output_path), latest_annotated)
                log_info(f"Last preview saved: {output_path}")
            except Exception as exc:  # noqa: BLE001
                log_warn(f"could not save the last preview: {exc}")

        try:
            if perf.total("capture") > 0:
                log_perf_summary()
        except Exception as exc:  # noqa: BLE001
            log_warn(f"could not print the performance summary: {exc}")
        if csv_handle is not None:
            try:
                csv_handle.close()
            except OSError:
                pass

        # Final, unambiguous account of what happened and in what order.
        log_info(f"Shutdown reason (first recorded): {SHUTDOWN.reason}")
        for worker in workers.values():
            log_info(
                f"  {worker.name} worker PID {worker.pid}: exit code {worker.exit_code}; "
                f"{worker.exit_class or 'still running at cleanup'}; stop: {worker.stop_how}"
            )
        if SHUTDOWN.first_failure:
            log_error(f"First failure of this session: {SHUTDOWN.first_failure}")
        log_info(f"Diagnostic logs: {WORKER_LOG_DIR} (files starting with {SESSION_TAG})")
        log_info("Live test ended.")


# ============================================================
# CLI
# ============================================================
def parse_threads_option(text):
    """'4,2,3' -> {'rf': 4, 'lane': 2, 'metric': 3}"""
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated integers: rf,lane,metric (for example 4,2,3)")
    values = [int(piece) for piece in parts]
    if any(value < 1 or value > 32 for value in values):
        raise ValueError("thread counts must be between 1 and 32")
    return {"rf": values[0], "lane": values[1], "metric": values[2]}


def main():
    parser = argparse.ArgumentParser(
        description="VISENTRA: RF-DETR object detection + Metric3D v2 depth + CurveLaneS lanes + "
        "ByteTrack (+ optional DeepSORT / MobileNetV3)"
    )
    parser.add_argument("--image", default=None, help="Input image path for still-image mode")
    parser.add_argument("--worker", choices=["rf", "lane", "metric"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", default=None, help="Worker result JSON path")
    parser.add_argument("--detections-file", default=None,
                        help="RF-DETR JSON input for the Metric3D worker")
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE,
                        help="Provisional raw Metric3D depth multiplier; default 0.373572")
    parser.add_argument("--live", action="store_true",
                        help="Run live inference on the Vivo rear camera (started automatically)")
    parser.add_argument("--video-dir", default=None,
                        help="Play every supported recorded video in this folder through the existing "
                             "VISENTRA pipeline at the original frame rate")
    parser.add_argument("--camera", type=int, default=0,
                        help="Ignored in live mode (kept for compatibility): the Vivo camera "
                             "is read directly, no webcam index is used")
    parser.add_argument("--serial", default=None,
                        help="adb serial of the phone (only needed if several devices are connected)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15,
                        help="Requested Vivo camera frame rate (default 15)")
    parser.add_argument("--worker-stream", choices=["rf", "lane", "metric"], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--test-known-distances", action="store_true",
                        help="Run RF-DETR + Metric3D on the existing 5m/10m/15m/20m images")
    parser.add_argument("--select-targets", action="store_true",
                        help="Show local numbered detections and let you choose the measured target vehicle in each test image")
    parser.add_argument("--with-lanes", action="store_true",
                        help="Also run CurveLaneS (live mode and --test-known-distances)")
    parser.add_argument("--skip-lanes", action="store_true",
                        help="Skip CurveLaneS (still-image mode; in live mode lanes are off unless --with-lanes)")
    # ---- performance options ----
    parser.add_argument("--threads", default=None, metavar="RF,LANE,METRIC",
                        help="CPU threads per worker, e.g. 4,2,3 (default 4,2,3). Not benchmarked: "
                             "compare the measured FPS/latency of different settings on the same video.")
    parser.add_argument("--lane-every", type=int, default=1, metavar="N",
                        help="Send every N-th completed RF-DETR frame to CurveLaneS (default 1)")
    parser.add_argument("--metric-every", type=int, default=1, metavar="N",
                        help="Send every N-th completed RF-DETR frame to Metric3D (default 1)")
    parser.add_argument("--with-deepsort", action="store_true",
                        help="Also run the DeepSORT comparison tracker (off by default; CPU heavy)")
    parser.add_argument("--with-mobilenet", action="store_true",
                        help="Also run MobileNetV3 crop classification (off by default; CPU heavy)")
    parser.add_argument("--detection-max-age", type=float, default=DETECTION_STALE_SECONDS,
                        help="Do not draw RF-DETR results captured more than this many seconds ago (default 5)")
    parser.add_argument("--lane-max-age", type=float, default=LANE_STALE_SECONDS,
                        help="Do not draw lane results captured more than this many seconds ago (default 8)")
    args = parser.parse_args()

    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("--depth-scale must be a positive finite number")

    if args.worker_stream is not None:
        worker_stream(args.worker_stream, args.depth_scale)
        return

    if args.worker is not None:
        if args.image is None:
            parser.error("--image is required in worker mode")
        if args.result is None:
            parser.error("--result is required in worker mode")
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        if args.worker == "rf":
            worker_rf(image_path, args.result)
        elif args.worker == "lane":
            worker_lane(image_path, args.result)
        else:
            if args.detections_file is None:
                parser.error("--detections-file is required for Metric3D worker mode")
            worker_metric(image_path, args.detections_file, args.result, args.depth_scale)
        return

    # From here on this is the parent process: apply the optional-model switches.
    set_mobilenet_enabled(args.with_mobilenet)
    try:
        thread_config = parse_threads_option(args.threads) if args.threads else dict(DEFAULT_THREADS)
    except ValueError as exc:
        parser.error(f"--threads: {exc}")
    if args.lane_every < 1 or args.metric_every < 1:
        parser.error("--lane-every and --metric-every must be 1 or larger")
    if args.detection_max_age <= 0 or args.lane_max_age <= 0:
        parser.error("--detection-max-age and --lane-max-age must be positive")

    if args.select_targets and not args.test_known_distances:
        parser.error("--select-targets can only be used with --test-known-distances")

    if args.test_known_distances:
        test_known_distances(
            depth_scale=args.depth_scale,
            with_lanes=args.with_lanes,
            select_targets=args.select_targets,
        )
        return

    live_options = dict(
        depth_scale=args.depth_scale,
        with_lanes=args.with_lanes,
        threads=thread_config,
        lane_every=args.lane_every,
        metric_every=args.metric_every,
        enable_deepsort=args.with_deepsort,
        detection_max_age=args.detection_max_age,
        lane_max_age=args.lane_max_age,
    )

    if args.video_dir is not None:
        if args.live:
            parser.error("--video-dir and --live cannot be used together")
        if args.with_lanes and args.skip_lanes:
            parser.error("--with-lanes and --skip-lanes cannot be used together")
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.is_dir():
            parser.error(f"--video-dir is not a folder: {video_dir}")
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv"}
        videos = sorted(
            (p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions),
            key=lambda p: p.name.lower()
        )
        if not videos:
            parser.error(f"No supported video files found in {video_dir}")
        log_info(f"Found {len(videos)} recorded video(s) in {video_dir}")
        for index, video_path in enumerate(videos, start=1):
            print(f"\n========== VIDEO {index}/{len(videos)}: {video_path.name} ==========", flush=True)
            try:
                master_live(video_path=video_path, **live_options)
            except (RuntimeError, FileNotFoundError, TimeoutError) as exc:
                log_error(f"Could not process {video_path.name}: {exc}")
                continue
            if SHUTDOWN.reason and "user pressed Q/ESC" in SHUTDOWN.reason:
                log_info("Playback stopped by user; remaining videos will be skipped.")
                break
        return

    if args.live:
        if args.width <= 0 or args.height <= 0:
            parser.error("--width and --height must be positive integers")
        if args.camera_fps <= 0:
            parser.error("--camera-fps must be a positive integer")
        if args.with_lanes and args.skip_lanes:
            parser.error("--with-lanes and --skip-lanes cannot be used together")
        try:
            master_live(
                camera_index=args.camera, width=args.width, height=args.height,
                serial=args.serial, camera_fps=args.camera_fps, **live_options,
            )
        except VivoCameraError as exc:
            log_error(str(exc))
            sys.exit(2)
        except (RuntimeError, FileNotFoundError, TimeoutError) as exc:
            log_error(str(exc))
            sys.exit(1)
        except KeyboardInterrupt:
            log_info("Interrupted by the user.")
        return

    if args.image is None:
        parser.error("Use --test-known-distances, --live, --video-dir, or --image for still-image mode")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    master(image_path, depth_scale=args.depth_scale, run_lanes=not args.skip_lanes)


if __name__ == "__main__":
    main()
