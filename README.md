# Lane Change Counting

Counts vehicle lane changes in fixed-camera traffic videos with one pipeline and the same settings for every video. No video-specific values, no per-video logic, and no vision-language model.

## Setup

```bash
uv sync
uv run python run.py                      # all lane_change_count_*.mp4 in this folder
uv run python run.py a.mp4 b.mp4 --out outputs --answer answer.json
```

- Developed and tested on Linux with an NVIDIA H200 (CUDA 12.8). `pyproject.toml` pins the CUDA 12.8 build of PyTorch on Linux. On other platforms, `uv` installs the default PyTorch build, which runs on CPU and is much slower. That path is untested.
- `yolov8l.pt` is downloaded automatically on first run, so the first run needs internet access.
- Python 3.13 or newer.
- The input videos go in the same folder as `run.py`, or are passed as arguments.
- Output: `answer.json`, plus `outputs/<video>_annotated.mp4` (H.264) and `outputs/<video>_events.json` for each video.

## Approach

1. **Detect and track vehicles** with YOLOv8l and ByteTrack at 1280 px inference size. Each vehicle's position is the bottom-center of its box, which is closest to where it touches the road.
2. **Infer lanes from traffic**, not from paint. At many image rows, histogram where tracked vehicles cross that row and take the peaks as lane centers. The centers are fitted as curves of image row. Tracks that barely move (parked cars) are excluded. I tried painted-line detection first (Canny, Hough, top-hat), but the road had curbs, a barrier, glare and few painted dividers, and it didn't give usable lanes.
3. **Assign each vehicle to a lane** at every frame, with hysteresis so jitter near a boundary doesn't flip lanes.
4. **Count a lane change** when a vehicle holds one lane, then holds an adjacent lane for at least 0.5 s.

## Annotated videos

Each `outputs/<video>_annotated.mp4` shows the fitted lane centers (yellow), a dot and `track id:lane` label on every tracked vehicle (gray `-` means unassigned), a red circle for 2 s at each counted lane change, and a running count at the top left. They are produced by the pipeline and are not edited by hand.

## Assumptions and limits

- **Extrapolated top zone.** Lane centers are measured from about 10% down the frame (the first fitted row). Above that they are straight-line continuations. Nearly all counted events are in that zone, where lanes are only a few percent of the frame width apart.
- **Barrier test is a heuristic.** Two lanes are treated as separate roads when the gap between them is more than 1.4 times the narrowest gap at the same height, judged no higher than the first fitted row. That cutoff is a guessed value, not derived. The gap ratios measured on the three videos spanned roughly 1.0 to 2.4.
- **Cars leaving the bottom edge.** If a track ends at the bottom edge of the frame, a final lane run of at least 0.2 s is accepted instead of 0.5 s.
- **Short tracks are dropped.** Tracks under 8 points are treated as flicker. Cars never detected, or tracked for only a few frames, can't be counted.
- **Merges count.** A vehicle moving from the right road into the left road near the top of the frame is counted as a lane change. Whether that fits the task's definition is a judgment call.
- **Fixed camera.** The method assumes a fixed camera and traffic moving in one direction.
- **Checking.** Counts were checked by eye against the annotated videos. I did not have reference counts to compare against.
- **Determinism.** Two runs on the same machine, and a fresh install from `uv.lock`, gave identical events on all three videos. Other GPUs may give slightly different detections.

## Results

| Video | Lane changes |
|---|---|
| lane_change_count_1.mp4 | 4 |
| lane_change_count_2.mp4 | 7 |
| lane_change_count_3.mp4 | 4 |