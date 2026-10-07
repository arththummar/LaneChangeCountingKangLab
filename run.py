"""Lane-change counting. Usage:
    uv run python run.py                       # all lane_change_count_*.mp4 in this folder
    uv run python run.py a.mp4 b.mp4 --out outputs --answer answer.json
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

WEIGHTS = "yolov8l.pt"                   # downloaded automatically on first use
IMGSZ = 1280                             # larger inference size so distant cars are detected
VEHICLE_CLASSES = [2, 3, 5, 7]           # COCO: car, motorcycle, bus, truck
MIN_TRACK_POINTS = 8                     # shorter tracks are flicker
MIN_TRAVEL_CARS = 1.0                    # must travel this many of its own car-heights
ROW_FRACS = np.linspace(0.10, 0.90, 33)  # image rows used to locate lane bands
SIGMA_FRAC = 0.005                       # histogram smoothing, fraction of frame width
SPLIT_RATIO = 1.4                        # gap this much wider than the narrowest = barrier between lanes
MIN_STABLE_SEC = 0.5                     # a lane must be held this long to count
OFF_LANE_FRAC = 0.75                     # beyond the outermost lane centers, farther than this = unassigned
HYST_FRAC = 0.2                          # must be this far past the midline to switch lanes
EXIT_MIN_SEC = 0.2                       # shortest final run accepted when a car drives out the bottom
EDGE_FRAC = 0.03                         # track must end within this fraction of frame height of the bottom


def track_video(video):
    model = YOLO(WEIGHTS)                # fresh model per video so tracker state never carries over
    raw = {}
    stream = model.track(source=str(video), stream=True, persist=True, tracker="bytetrack.yaml",
                         classes=VEHICLE_CLASSES, imgsz=IMGSZ, verbose=False)
    for f, r in enumerate(stream):
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
            # bottom-center of the box is closest to where the car touches the road
            raw.setdefault(tid, []).append((f, float((x1 + x2) / 2), float(y2), float(y2 - y1)))
    return {tid: np.array(p, dtype=np.float32) for tid, p in raw.items()}


def split_tracks(raw):
    all_tracks, kept = {}, {}
    for tid, a in raw.items():
        if len(a) < MIN_TRACK_POINTS:
            continue
        all_tracks[tid] = a
        travel = np.hypot(a[-1, 1] - a[0, 1], a[-1, 2] - a[0, 2])
        if travel >= MIN_TRAVEL_CARS * np.median(a[:, 3]):
            kept[tid] = a
    return all_tracks, kept


class LaneModel:
    """Lane bands inferred from where tracked vehicles drive, then lane assignment and change counting."""

    def __init__(self, tracks, w, h, fps):
        self.w, self.h, self.fps = w, h, fps
        rows, peaks = [], []
        for f in ROW_FRACS:
            r = int(f * h)
            p = self._peaks_at(tracks, r)
            if p is not None and len(p) > 0:
                rows.append(r)
                peaks.append(p)
        if not rows:
            raise RuntimeError("No lane bands found")
        self.K = int(np.bincount([len(p) for p in peaks]).argmax())
        good = [(r, p) for r, p in zip(rows, peaks) if len(p) == self.K]
        self.stable_rows, self.total_rows = len(good), len(rows)
        if self.K < 2 or len(good) < 3:
            raise RuntimeError("Band count is not stable enough to fit lanes")
        R = np.array([r for r, _ in good], dtype=float)
        X = np.stack([p for _, p in good])
        deg = 2 if len(R) >= 4 else 1
        self.coefs = [np.polyfit(R, X[:, k], deg) for k in range(self.K)]
        self.r_min, self.r_max = R.min(), R.max()
        self.slope_lo = [np.polyval(np.polyder(c), self.r_min) for c in self.coefs]
        self.slope_hi = [np.polyval(np.polyder(c), self.r_max) for c in self.coefs]
        self.y_lo, self.y_hi = 0.0, 0.97 * h

    @staticmethod
    def _x_at_row(a, row):
        ys, xs = a[:, 2], a[:, 1]
        s = np.sign(ys - row)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        t = (row - ys[i]) / (ys[i + 1] - ys[i])
        return float(xs[i] + t * (xs[i + 1] - xs[i]))

    def _peaks_at(self, tracks, row):
        xs = [x for x in (self._x_at_row(a, row) for a in tracks.values()) if x is not None]
        if len(xs) < 5:
            return None
        hist, _ = np.histogram(xs, bins=self.w, range=(0, self.w))
        hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), SIGMA_FRAC * self.w).ravel()
        p, _ = find_peaks(hist, height=0.2 * hist.max(), distance=int(0.008 * self.w))
        return p.astype(float)

    def centers_at(self, y):
        out = []
        for k, c in enumerate(self.coefs):
            if y < self.r_min:      # continue straight up from the edge of the fit
                out.append(np.polyval(c, self.r_min) + self.slope_lo[k] * (y - self.r_min))
            elif y > self.r_max:    # and straight down
                out.append(np.polyval(c, self.r_max) + self.slope_hi[k] * (y - self.r_max))
            else:
                out.append(np.polyval(c, y))
        return np.array(out)

    def same_road(self, k1, k2, y):
        """No unusually wide gap between the two bands, judged no higher than the first fitted row."""
        g = np.abs(np.diff(self.centers_at(max(float(y), self.r_min))))
        lo, hi = sorted((k1, k2))
        return g[lo:hi].max() / max(g.min(), 1e-6) < SPLIT_RATIO

    def label_track(self, a):
        labels, cur = [], -1
        for _, x, y, _ in a:
            if y < self.y_lo or y > self.y_hi:
                labels.append(-1)
                continue
            c = self.centers_at(y)
            if x < c[0] and c[0] - x > OFF_LANE_FRAC * (c[1] - c[0]):
                labels.append(-1)
                continue
            if x > c[-1] and x - c[-1] > OFF_LANE_FRAC * (c[-1] - c[-2]):
                labels.append(-1)
                continue
            k = int(np.argmin(np.abs(c - x)))
            if cur == -1:
                cur = k
            elif k != cur:
                gap = abs(c[k] - c[cur])
                if abs(x - c[cur]) - abs(x - c[k]) > HYST_FRAC * gap:
                    cur = k
            labels.append(cur)
        return labels

    def stable_runs(self, labels, a):
        min_frames = max(3, int(MIN_STABLE_SEC * self.fps))
        exit_frames = max(3, int(EXIT_MIN_SEC * self.fps))
        runs, s = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[s]:
                runs.append((labels[s], s, i - 1))
                s = i
        ends_at_bottom = a[-1, 2] >= (1 - EDGE_FRAC) * self.h
        last_lab = max((i for i, r in enumerate(runs) if r[0] != -1), default=None)
        out = []
        for i, r in enumerate(runs):
            if r[0] == -1:
                continue
            n = r[2] - r[1] + 1
            if n >= min_frames or (i == last_lab and ends_at_bottom and n >= exit_frames):
                out.append(r)
        return out

    def events(self, tracks):
        out = []
        for tid, a in tracks.items():
            stable = self.stable_runs(self.label_track(a), a)
            for prev, cur in zip(stable, stable[1:]):
                if prev[0] != cur[0] and self.same_road(prev[0], cur[0], a[cur[1], 2]):
                    out.append({"track": int(tid), "from_lane": prev[0], "to_lane": cur[0],
                                "frame": int(a[cur[1], 0]), "x": float(a[cur[1], 1]),
                                "y": float(a[cur[1], 2])})
        return sorted(out, key=lambda e: e["frame"])


class H264Writer:
    """H.264 + yuv420p through imageio-ffmpeg's bundled binary, so any reviewer machine can play it."""

    def __init__(self, path, w, h, fps):
        try:
            import imageio_ffmpeg
            self.ff = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, pix_fmt_in="bgr24",
                                                  pix_fmt_out="yuv420p", codec="libx264",
                                                  macro_block_size=1, quality=7)
            self.ff.send(None)
            self.cv = None
        except ImportError:
            print("WARNING: imageio-ffmpeg not installed, writing mp4v (may not play in browsers)")
            self.ff = None
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.ff is not None:
            self.ff.send(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.ff is not None:
            self.ff.close()
        else:
            self.cv.release()


def write_annotated(video, path, model, all_tracks, events, w, h, fps):
    per_frame = {}
    for tid, a in all_tracks.items():
        for (f_, x_, y_, _), lab in zip(a, model.label_track(a)):
            per_frame.setdefault(int(f_), []).append((tid, x_, y_, lab))
    ev_by_frame = {}
    for e in events:
        ev_by_frame.setdefault(e["frame"], []).append(e)
    ys = np.arange(0, int(0.95 * h), 4)
    curves = [np.array([[model.centers_at(y)[k], y] for y in ys], dtype=np.int32).reshape(-1, 1, 2)
              for k in range(model.K)]
    palette = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255)]
    cap = cv2.VideoCapture(str(video))
    writer = H264Writer(path, w, h, fps)
    count, recent, fi = 0, [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for pts in curves:
            cv2.polylines(frame, [pts], False, (0, 255, 255), 1)
        for tid, x_, y_, lab in per_frame.get(fi, []):
            col = palette[lab % len(palette)] if lab >= 0 else (160, 160, 160)
            cv2.circle(frame, (int(x_), int(y_)), 4, col, -1)
            cv2.putText(frame, f"{tid}:{lab if lab >= 0 else '-'}", (int(x_) + 5, int(y_) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        for e in ev_by_frame.get(fi, []):
            count += 1
            recent.append([e["x"], e["y"], int(2 * fps), e["track"]])
        for r in recent:
            cv2.circle(frame, (int(r[0]), int(r[1])), 18, (0, 0, 255), 2)
            cv2.putText(frame, f"lane change, track {r[3]}", (int(r[0]) + 20, int(r[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            r[2] -= 1
        recent = [r for r in recent if r[2] > 0]
        cv2.putText(frame, f"lane changes: {count}   frame {fi}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.close()


def process(video, out_dir):
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"\n=== {video.name}: {w}x{h}, {fps:.1f} fps ===")

    raw = track_video(video)
    all_tracks, kept = split_tracks(raw)
    print(f"{len(raw)} tracks, {len(all_tracks)} with >= {MIN_TRACK_POINTS} points, {len(kept)} kept after travel filter")

    model = LaneModel(kept, w, h, fps)
    print(f"{model.K} bands, stable at {model.stable_rows} of {model.total_rows} rows, "
          f"fitted rows {model.r_min / h:.2f}..{model.r_max / h:.2f}")
    events = model.events(kept)
    print(f"Lane changes: {len(events)}")
    for e in events:
        print(f"  track {e['track']}: lane {e['from_lane']} -> {e['to_lane']} "
              f"at {e['frame'] / fps:.1f}s (frame {e['frame']}), y/h = {e['y'] / h:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(events, open(out_dir / f"{video.stem}_events.json", "w"), indent=2)
    write_annotated(video, out_dir / f"{video.stem}_annotated.mp4", model, all_tracks, events, w, h, fps)
    return len(events)


def main():
    ap = argparse.ArgumentParser(description="Count vehicle lane changes in traffic videos.")
    ap.add_argument("videos", nargs="*", help="input videos (default: lane_change_count_*.mp4 here)")
    ap.add_argument("--out", default="outputs", help="folder for annotated videos and event lists")
    ap.add_argument("--answer", default="answer.json", help="where to write the counts")
    args = ap.parse_args()

    videos = args.videos or sorted(v for v in glob.glob("lane[-_]change[-_]count[-_]*.mp4")
                                   if "annotated" not in v)
    if not videos:
        raise SystemExit("No input videos found")
    answer = {}
    for v in videos:
        answer[Path(v).name.replace("-", "_")] = {"total_lane_changes": process(v, Path(args.out))}
    json.dump(answer, open(args.answer, "w"), indent=2)
    print(f"\nWrote {args.answer}: {answer}")


if __name__ == "__main__":
    main()