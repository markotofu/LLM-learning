"""Chart data for the sample lesson game-loop: distance covered in one second at 30, 60 and 144 fps by an object that
moves a fixed 0.1 units per frame, and by one that moves 5 units per second scaled by delta time.
Prints the chart spec as JSON.   python tools/charts/game_loop.py"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")   # the legend has a Δ


def run(fps, seconds=1.0, step=0.1, speed=5.0):
    frames = round(fps * seconds)
    dt = 1 / fps
    per_frame = timed = 0.0
    for _ in range(frames):
        per_frame += step
        timed += speed * dt
    return round(per_frame, 2), round(timed, 2)


rates = [30, 60, 144]
results = [run(f) for f in rates]
spec = {
    "kind": "bar",
    "x": [f"{f} fps" for f in rates],
    "series": [
        {"name": "Fixed 0.1 units per frame", "y": [r[0] for r in results]},
        {"name": "5 units per second × Δt", "y": [r[1] for r in results]},
    ],
    "xLabel": "Frame rate",
    "yLabel": "Distance in one second (units)",
}
print(json.dumps(spec, ensure_ascii=False))
