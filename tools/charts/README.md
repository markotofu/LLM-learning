# Chart data scripts

Every number in a lesson chart comes from a real computation (a simulation, a fitted model or a formula), never from
invented values. Keep the script that computed a chart here, so the numbers can be checked and regenerated.

```bash
python tools/charts/game_loop.py      # prints the spec of the chart in the sample lesson game-loop
```

| Script | Lesson | Chart |
| --- | --- | --- |
| `game_loop.py` | `game-loop` | Distance covered in one second at three frame rates |

A script can print the chart spec as JSON for you to paste into the lesson, or write a lesson patch with an
`### INSERT … @@ chart` block for `node tools/lesson-patch.js` (see docs/CONTENT_GUIDE.md).
