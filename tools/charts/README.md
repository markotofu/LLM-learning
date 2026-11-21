# Chart data scripts

These scripts computed the numbers behind the 155 lesson charts: every curve, point and bar comes from a real
computation (a simulation, a fitted model or a formula), never from invented values. Each script covers one batch of
lessons and writes a lesson patch with one `### INSERT … @@ chart` block per chart:

```bash
python tools/charts/charts_n2.py out.txt      # needs numpy, pandas and scikit-learn
```

| Script | Lessons |
| --- | --- |
| `charts_m1.py`, `charts_m2.py` | maths foundations, Python and NumPy |
| `charts_batch1.py`, `charts_c1.py`–`charts_c3.py` | classical machine learning, unsupervised learning, applied lessons |
| `charts_n1.py`–`charts_n3.py` | neural networks and deep learning |
| `charts_l1.py`–`charts_l5.py` | transformers and LLMs |
| `charts_a1.py` | building with LLMs (prompting, RAG, vector search, evaluation, agents) |
| `charts_s1.py`, `charts_s2.py` | specializations, reinforcement learning, MLOps, responsible AI and ML security |
| `charts_x1.py`–`charts_x4.py` | Advanced Agents and the three capstones (tabular, RAG, LoRA) |
| `charts_y1.py` | the lessons added for the roadmap.sh roadmaps: video models, world models, robotics & VLAs, AI engineering, LLMs in production, discrete maths |
| `charts_z1.py` | the ten topic lessons: graphical models, GLMs and survival analysis, 3D vision, AI for science, audio generation, AI governance, contextual bandits, interpretability in practice, AI hardware |

`charts_x3.py`, `charts_x4.py`, `charts_y1.py` and `charts_z1.py` read the code of the samples they chart straight from the lesson (through Node, so run
them from a checkout with Node installed) and execute it, so their charts always match what the samples print.

The lesson files in `js/data/` are the source of truth. The charts are already in them, and some captions were reworded
after they were generated, so do not apply these patches again (each chart would be inserted twice). Use a script to
check or recompute a chart's numbers, for example after changing the experiment, and replace that chart's `chart`
spec in the lesson by hand. The `Apply AFTER patches/…` notes refer to the text patches that added the sections these
charts are anchored to; those patches are already part of the lessons.
