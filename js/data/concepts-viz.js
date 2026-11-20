/* ML Atlas – content: data visualization with matplotlib and pandas */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'data-visualization', branch: 'code', title: 'Data Visualization with matplotlib & pandas', short: 'Data Visualization', level: 1, minutes: 50, prereqs: ['pandas-eda'],
    tags: ['visualization', 'matplotlib', 'pandas', 'plot', 'figure', 'axes', 'subplots', 'histogram', 'scatter', 'bar chart', 'density', 'kde', 'scatter matrix', 'seaborn', 'eda'],
    summary: 'Turning data into pictures that reveal what summary numbers hide: the matplotlib figure and axes API, styles, labels, annotations and saving; plotting straight from pandas (line, bar, histogram, density and scatter plots, scatter matrices); facets with seaborn; choosing the right chart; and the diagnostic plots of machine learning.',
    why: `Two datasets can share the same mean, variance and correlation and still look nothing alike. A plot shows outliers, clusters, curves, gaps and data-entry errors in seconds, which is why every analysis and every modelling project should start by **looking** at the data, and end by showing the results in a way others can trust. Visualization is also how you debug models: learning curves, confusion matrices and residual plots tell you what went wrong far faster than a table of numbers.`,
    objectives: [
      'Build figures with several subplots using the matplotlib figure and axes API',
      'Control colours, markers, line styles, ticks, labels, legends and annotations, and save figures to files',
      'Make line, bar, histogram, density and scatter plots directly from pandas objects',
      'Choose a chart that fits the question, and use plots to explore data and diagnose models'
    ],
    sections: [
      { type: 'text', title: "Why plot at all?", md: `Summary statistics compress a dataset into a few numbers, and compression loses information. A plot keeps the shape: clusters, curves, gaps, outliers, ceilings, typos. That is why experienced analysts plot **before** they model, and why a model's results are best checked and explained with pictures too.

Plots do three jobs in machine learning:

1. **Exploration** — what is in the data? Are there errors, outliers, unexpected groups? Do the features separate the classes at all?
2. **Diagnosis** — is the model overfitting (learning curves)? Which classes does it confuse (confusion matrix)? Where are its errors largest (residual plots)?
3. **Communication** — a single clear chart convinces a colleague or a manager faster than a table.

The standard Python tools are **matplotlib**, the foundation everything else is built on, **pandas' plotting methods**, which call matplotlib with sensible defaults, and **seaborn**, which adds statistical charts.` },
      { type: 'code', title: "Same statistics, different stories: Anscombe's quartet", md: `In 1973 the statistician Francis Anscombe built four small datasets with nearly identical means, variances, correlations and regression lines. Only plotting them reveals how different they are.`, code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
sets = {
    "I":   (x, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II":  (x, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (x, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV":  ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
rows = []
for name, (xs, ys) in sets.items():
    slope, intercept = np.polyfit(xs, ys, 1)
    rows.append({"set": name, "mean x": np.mean(xs), "mean y": np.mean(ys), "var y": np.var(ys, ddof=1),
                 "corr": np.corrcoef(xs, ys)[0, 1], "line": f"y = {intercept:.2f} + {slope:.3f}x"})
print(pd.DataFrame(rows).set_index("set").round(2))
print("correlations:", [round(r["corr"], 2) for r in rows])

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
for ax, (name, (xs, ys)) in zip(axes.ravel(), sets.items()):
    ax.scatter(xs, ys, color="tab:blue")
    grid = np.array([3, 20])
    ax.plot(grid, 3 + 0.5 * grid, color="tab:orange")      # the shared regression line
    ax.set_title(f"dataset {name}")
fig.suptitle("Four datasets, one set of summary statistics")
fig.tight_layout()
plt.show()`, after: `Dataset I is the textbook case: a noisy straight line. II is a perfect **curve**, which a straight line describes badly. III is a perfect line spoiled by **one outlier**, which tilts the fit. In IV, all but one point share the same x: the "relationship" exists only because of **one extreme point**. A correlation of 0.82 hides all of this. Always look.`, predict: {"type":"mcq","q":"The four datasets share (almost exactly) the same means, variances, correlation (0.82) and regression line. What will their scatter plots look like?","options":["Four nearly identical clouds of points","Four very different shapes: a noisy line, a curve, a line with an outlier, and a vertical stack with one far point","Four horizontal lines","Four random clouds with no pattern"],"answer":1,"explain":"Identical summary statistics can come from completely different data. Only the plots reveal the curve, the outlier and the single influential point.","expect":["correlations: [0.82, 0.82, 0.82, 0.82]"]} },
      { type: 'code', title: "Figures, axes and subplots", md: `matplotlib's model has two main objects. A **Figure** is the whole image; an **Axes** is one plotting area inside it, with its own x and y axes. \`plt.subplots(nrows, ncols)\` creates both at once and returns the Axes as an array shaped like the grid. You then call drawing methods on each Axes.`, code: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
print("axes array shape:", axes.shape)

axes[0, 0].plot(rng.standard_normal(100).cumsum(), "k--")              # a random walk, black dashed
axes[0, 0].set_title("line: plot")
axes[0, 1].hist(rng.standard_normal(500), bins=20, color="tab:blue", alpha=0.7)
axes[0, 1].set_title("distribution: hist")
x = np.arange(30)
axes[1, 0].scatter(x, x + 3 * rng.standard_normal(30), color="tab:green")
axes[1, 0].set_title("relationship: scatter")
axes[1, 1].bar(["A", "B", "C", "D"], [5, 3, 7, 2], color="tab:orange")
axes[1, 1].set_title("comparison: bar")

fig.suptitle("One figure, four axes")
fig.tight_layout()                                                     # stop titles and labels overlapping
plt.show()`, after: `Useful options of \`plt.subplots\`:

| Argument | Meaning |
| --- | --- |
| \`nrows\`, \`ncols\` | the grid of subplots |
| \`sharex\`, \`sharey\` | all subplots use the same x (or y) limits and ticks, which makes them comparable |
| \`figsize\` | the figure size in inches, e.g. \`(8, 6)\` |
| \`subplot_kw\` | extra settings passed to every subplot (for example a polar projection) |

\`fig.tight_layout()\` (or \`plt.subplots(..., layout="constrained")\`) adjusts the spacing so titles and tick labels do not collide; \`fig.subplots_adjust(wspace=..., hspace=...)\` sets the gaps by hand.

You will also see the shorter **pyplot style**, \`plt.plot(...)\`, \`plt.title(...)\`, which draws on the "current" Axes. It is fine for one quick plot; the explicit \`fig, ax\` style is clearer as soon as there is more than one subplot.` },
      { type: 'text', title: "Colours, markers and line styles", md: `\`ax.plot(x, y, "go--")\` uses a **format string**: a colour letter, a marker and a line style. The same can be spelled out with keywords, which is easier to read:

\`\`\`python
ax.plot(x, y, color="g", marker="o", linestyle="--", linewidth=2, alpha=0.8, label="model A")
\`\`\`

| Kind | Common values |
| --- | --- |
| colour | \`"k"\` black, \`"b"\` blue, \`"r"\` red, \`"g"\` green; names such as \`"tab:orange"\`; hex codes such as \`"#1f77b4"\` |
| marker | \`"o"\` circle, \`"s"\` square, \`"^"\` triangle, \`"."\` small point, \`"x"\` cross |
| line style | \`"-"\` solid, \`"--"\` dashed, \`":"\` dotted, \`"-."\` dash-dot |
| other | \`linewidth\`, \`markersize\`, \`alpha\` (transparency from 0 to 1), \`drawstyle="steps-post"\` for step functions |

For colour scales, prefer **perceptually uniform** colormaps such as \`"viridis"\`: they read correctly in greyscale and for colour-blind readers, unlike red-to-green scales.` },
      { type: 'code', title: "Titles, labels, ticks, legends and annotations", md: `A chart is only as good as its labelling. Every Axes method below has a \`set_…\` form; \`ax.set(title=..., xlabel=...)\` sets several at once.`, code: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
days = np.arange(60)
visits = 200 + days * 3 + 25 * rng.standard_normal(60)
visits[40] += 180                                            # a one-day spike
baseline = 200 + days * 3

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(days, visits, color="tab:blue", label="daily visits")
ax.plot(days, baseline, "k--", linewidth=1, label="trend")
ax.axhline(visits.mean(), color="grey", linestyle=":", label=f"mean ({visits.mean():.0f})")
ax.set(title="Website visits over two months", xlabel="day", ylabel="visits per day")
ax.set_xticks([0, 15, 30, 45, 59])
ax.set_xticklabels(["1 Mar", "16 Mar", "31 Mar", "15 Apr", "29 Apr"], rotation=30)
ax.annotate("newsletter sent", xy=(40, visits[40]), xytext=(15, visits[40] + 10),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.axvspan(50, 59, color="tab:orange", alpha=0.15)           # shade a period of interest
ax.text(51, visits.min(), "holiday", color="tab:orange")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
print("peak day:", int(np.argmax(visits)), " peak visits:", int(visits.max()))`, after: `- Give every curve a \`label=\` and call \`ax.legend()\`; \`loc="best"\` lets matplotlib choose a free corner.
- \`set_xticks\` chooses where ticks go and \`set_xticklabels\` what they say (rotate long labels).
- **\`annotate\`** points an arrow at a data point with a text label; \`text\` places free text; \`axhline\`, \`axvline\` and \`axvspan\` draw reference lines and shaded regions. Explaining the one thing the reader should notice turns a plot into an argument.` },
      { type: 'text', title: "Saving figures", md: `\`fig.savefig("visits.png")\` writes the figure to a file; the format comes from the extension.

| Argument | Meaning |
| --- | --- |
| \`fname\` | the file path; \`.png\` and \`.jpg\` give pixel images, \`.svg\` and \`.pdf\` give vector images that stay sharp at any size |
| \`dpi\` | resolution in dots per inch for pixel formats; 100 by default, 200–300 for print |
| \`bbox_inches="tight"\` | trim the empty margin around the figure |
| \`facecolor\`, \`transparent\` | the background colour, or a transparent background |

Use vector formats (SVG or PDF) for reports and slides, and PNG for the web. In this app, figures are displayed automatically after the code runs.` },
      { type: 'code', title: "Plotting straight from pandas", md: `Series and DataFrames have a \`.plot\` method that calls matplotlib for you: the index becomes the x axis, each column becomes a line (or a bar group), and legends and labels are filled in automatically. \`kind=\` or the \`.plot.<kind>()\` methods choose the chart: \`line\`, \`bar\`, \`barh\`, \`hist\`, \`density\`, \`scatter\`, \`box\`, \`area\`, \`pie\`.`, code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats    # density (KDE) plots use SciPy

rng = np.random.default_rng(2)
walks = pd.DataFrame(rng.standard_normal((100, 3)).cumsum(axis=0), columns=["A", "B", "C"],
                     index=pd.date_range("2026-01-01", periods=100))
party = pd.DataFrame({"1 person": [3, 2, 9, 6], "2 people": [39, 48, 70, 57], "3+ people": [10, 12, 40, 30]},
                     index=["Thu", "Fri", "Sat", "Sun"])
tips = pd.Series(np.concatenate([rng.normal(12, 3, 300), rng.normal(24, 4, 120)]))

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
walks.plot(ax=axes[0, 0], title="DataFrame.plot: one line per column")
party.plot.bar(ax=axes[0, 1], stacked=True, rot=0, title="stacked bars per day")
pct = party.div(party.sum(axis=1), axis=0) * 100
pct.plot.barh(ax=axes[1, 0], stacked=True, title="100% stacked: shares per day")
tips.plot.hist(ax=axes[1, 1], bins=30, density=True, alpha=0.5, title="histogram + density")
tips.plot.density(ax=axes[1, 1], color="k")
fig.tight_layout()
plt.show()
print("share of parties of 2 on Saturday:", round(pct.loc["Sat", "2 people"], 1), "%")`, after: `- A **line plot** is right for ordered data such as time; pandas uses the date index for the x axis.
- **Bar plots** compare categories; \`stacked=True\` shows composition. A **100% stacked** version (rows divided by their totals, which is a practice problem in this lesson) compares *shares* between groups of different sizes. \`value_counts().plot.bar()\` is the quickest way to see how often each category occurs.
- A **histogram** with \`density=True\` and a **density plot** (a kernel density estimate, KDE: a smooth curve built by adding a small bell curve at every data point) show the distribution. Here they reveal two groups of tips that a single mean (about 15) would hide.

Useful arguments: \`ax=\` (which subplot to draw on), \`title=\`, \`rot=\` (tick label rotation), \`logy=True\`, \`xlim=\`/\`ylim=\`, \`grid=True\`, \`subplots=True\` (one subplot per column), \`figsize=\`.` },
      { type: 'code', title: "Scatter plots and the scatter matrix", md: `A **scatter plot** shows the relationship between two numeric variables. A **scatter matrix** (or pair plot) shows every pair at once, with each variable's distribution on the diagonal: the classic first look at a new dataset.`, code: `import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal length", "sepal width", "petal length", "petal width"])

df.plot.scatter(x="petal length", y="petal width", c=iris.target, colormap="viridis",
                title="petals, coloured by species", figsize=(6, 4))
plt.show()

pd.plotting.scatter_matrix(df, c=iris.target, figsize=(8, 8), diagonal="hist", alpha=0.8, marker="o")
plt.suptitle("iris: every pair of measurements")
plt.show()
print(df.corr().round(2))`, after: `The petal measurements separate the three species almost perfectly, and petal length and width are strongly correlated (0.96), while sepal width barely relates to the others. A model trained on these data will lean on the petals. Seeing that **before** training lets you predict what the model should do and spot when it does something else.` },
      { type: 'text', title: "Facets and seaborn", md: `To compare one relationship across groups, draw **small multiples** (also called facets or trellis plots): a grid of small plots, one per group, sharing the same axes. Your eye compares their shapes far more easily than overlapping colours in one plot.

**seaborn** builds such statistical plots on top of matplotlib with one call each: \`histplot\`, \`kdeplot\`, \`boxplot\` and \`violinplot\` for distributions, \`scatterplot\` and \`regplot\` for relationships, \`barplot\` (with confidence intervals), \`heatmap\`, and the faceting functions \`relplot\`, \`displot\` and \`catplot\`. seaborn is not available in this app's in-browser Python, so this example is for running locally:

\`\`\`python
import seaborn as sns
tips = sns.load_dataset("tips")                  # restaurant bills and tips
sns.catplot(data=tips, x="day", y="tip", hue="smoker", col="time", kind="box")
sns.pairplot(tips, hue="time")                    # a scatter matrix, coloured by group
\`\`\`

\`col=\` and \`row=\` split the data into facets, and \`hue=\` colours by a category inside each facet.` },
      { type: 'check', question: {"type":"mcq","q":"You want to show how each region's sales split between three product lines, comparing regions of very different sizes. Which chart fits best?","options":["A line chart","A 100% stacked bar chart, one bar per region","A scatter plot","A histogram"],"answer":1,"explain":"A 100% stacked bar shows composition as shares, so regions of different sizes can be compared directly."} },
      { type: 'text', title: "Choosing the right chart", md: `Start from the question you want the reader to answer:

| Question | Chart |
| --- | --- |
| How do values compare across categories? | bar chart (horizontal if the labels are long), sorted by value |
| How are values distributed? | histogram, density plot; box or violin plot to compare groups |
| How does one variable relate to another? | scatter plot; add colour for a third variable |
| How do many variables relate? | scatter matrix, correlation heat map |
| How does something change over time? | line chart |
| What are the parts of a whole? | stacked bar chart; a pie only for two or three slices |
| How does a relationship differ between groups? | small multiples (facets) |` },
      { type: 'code', title: "Plots that diagnose a model", md: `Three of the most useful diagnostic plots in machine learning: a **validation curve** (score against a complexity setting), a **confusion matrix** drawn as a heat map, and a **decision boundary**, obtained by asking the model for a prediction at every point of a fine grid.`, code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ks = [1, 3, 5, 9, 15, 25, 45, 75, 150]
train_acc, test_acc = [], []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    train_acc.append(m.score(X_train, y_train)); test_acc.append(m.score(X_test, y_test))
best_k = ks[int(np.argmax(test_acc))]
model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(ks, train_acc, "o-", label="training")
axes[0].plot(ks, test_acc, "s-", label="test")
axes[0].set(xscale="log", xlabel="k (neighbours)", ylabel="accuracy", title="validation curve")
axes[0].legend()

cm = confusion_matrix(y_test, model.predict(X_test))
axes[1].imshow(cm, cmap="Blues")
for (i, j), n in np.ndenumerate(cm):
    axes[1].text(j, i, n, ha="center", va="center", color="white" if n > cm.max() / 2 else "black")
axes[1].set(xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="actual", title="confusion matrix")

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[2].contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", s=20)
axes[2].set(title=f"decision boundary, k = {best_k}")
fig.tight_layout()
plt.show()
print("best k on the test set:", best_k, " accuracy:", round(max(test_acc), 3))`, after: `- The **validation curve** shows the classic picture: training accuracy falls as k grows (simpler model) while test accuracy peaks in the middle. (Choosing k on the test set is only for illustration here; in practice use [cross-validation](#/concept/cross-validation).)
- The **confusion matrix** heat map makes the mistakes jump out; with many classes it is the fastest way to see which pairs are confused.
- The **decision boundary** is drawn by predicting every point of a 200 × 200 grid and colouring the regions with \`contourf\`: the same trick works for any classifier with two input features.` },
      { type: 'text', title: "Habits of honest, readable charts", md: `- **Label everything**: axis titles with units, a legend for every colour, a title that states the point ("Sales doubled after the redesign") rather than the topic ("Sales").
- **Start bar charts at zero**; line and scatter plots may zoom in, because they encode position, not length.
- **Show uncertainty** when you compare estimates: error bars, confidence bands, or the raw points.
- **Use colour for meaning**, not decoration; keep a category's colour the same across charts; prefer colour-blind-safe palettes.
- **Avoid chartjunk**: 3-D effects, heavy grid lines, many-slice pies, unnecessary decoration.
- **One message per chart.** If a chart needs a paragraph to explain it, split it into small multiples or several charts.` },
      { type: 'check', question: {"type":"tf","q":"Two datasets with the same mean, variance and correlation must look alike in a scatter plot.","answer":false,"explain":"Anscombe's quartet is the classic counterexample: identical summary statistics, completely different shapes."} }
    ],
    when: `- At the start of every project: histograms, scatter plots and a scatter matrix before any model.
- Whenever a summary statistic surprises you: plot the raw data behind it.
- To check a model: learning curves, residuals, confusion matrices, calibration curves.
- To communicate a result: one clear chart with labelled axes and a title that states the takeaway.`,
    pitfalls: `- Trusting summary statistics without plotting the data (see Anscombe's quartet).
- Bar charts whose value axis does not start at zero, which exaggerate differences.
- Unlabelled axes, missing units, and legends that do not say what the colours mean.
- Too much in one picture: many overlapping lines or a pie with a dozen slices. Use small multiples instead.
- Colours that colour-blind readers cannot tell apart (red against green); prefer palettes such as viridis.
- Forgetting that a histogram's shape depends on the number of bins: try several.`,
    takeaways: [
      'In matplotlib, a Figure holds one or more Axes; draw on an Axes with methods such as plot, hist, scatter and bar.',
      'pandas objects plot themselves (Series.plot, DataFrame.plot) using matplotlib, with sensible labels and legends.',
      'Histograms and density plots show distributions; scatter plots and scatter matrices show relationships.',
      'Pick the chart from the question: comparison, distribution, relationship, composition or change over time.',
      'Plot before you model, and plot again to diagnose the model.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which plot best shows the **distribution** of a single numeric variable?', options: ['A pie chart', 'A histogram or density plot', 'A line chart of the values in row order', 'A scatter plot against the row number'], answer: 1, explain: 'Histograms (counts per bin) and density plots (a smooth estimate) show where values concentrate, how spread out they are, and whether there are several peaks or outliers.' },
      { type: 'mcq', q: 'After `fig, axes = plt.subplots(2, 3)`, what is `axes`?', options: ['A single Axes', 'A 2 × 3 NumPy array of Axes objects', 'A list of 6 figures', 'The figure size'], answer: 1, explain: 'subplots returns the Figure and an array of Axes shaped like the grid, so axes[1, 2] is the subplot in the second row, third column.' },
      { type: 'mcq', q: 'In matplotlib, what does the format string `"ko--"` mean?', options: ['Black, circle markers, dashed line', 'Blue, diamond markers, dotted line', 'A thick solid line', 'Plot every other point'], answer: 0, explain: '"k" is black, "o" draws circle markers and "--" a dashed line. The same can be written as color="k", marker="o", linestyle="--".' },
      { type: 'tf', q: 'The value axis of a bar chart should normally start at zero.', answer: true, explain: 'Bars encode values by their length; cutting the axis exaggerates small differences. Line charts and scatter plots, which encode position, may zoom in.' },
      { type: 'mcq', q: 'What is a scatter matrix (pair plot) for?', options: ['Showing one variable over time', 'Showing every pairwise relationship among several numeric variables at once', 'Showing the parts of a whole', 'Saving a figure as a matrix'], answer: 1, explain: 'It draws a grid of scatter plots, one for every pair of variables, with each variable\'s distribution on the diagonal: a quick first look at a new dataset.' },
      { type: 'mcq', q: 'You need to compare the distribution of income across 12 regions. Which approach is clearest?', options: ['One histogram with 12 overlapping colours', 'A pie chart', 'Small multiples: a grid of 12 small histograms (or box plots) sharing the same axes', 'A 3-D bar chart'], answer: 2, explain: 'Small multiples (facets) with shared axes let the eye compare shapes directly without untangling overlapping colours.' }
    ],
    problems: [
      {
        id: 'histogram-counts', title: 'Histogram Counts', difficulty: 'easy',
        prompt: `A histogram counts how many values fall into each of several equal-width **bins**. Implement \`histogram_counts(values, bins, lo, hi)\` that returns a list of \`bins\` counts for the interval from \`lo\` to \`hi\`:

- the bins have equal width (hi − lo) / bins and cover the interval from left to right;
- a value on a boundary belongs to the bin on its right, **except** \`hi\` itself, which belongs to the last bin (this is how \`np.histogram\` works);
- values below \`lo\` or above \`hi\` are ignored.

\`\`\`text
histogram_counts([1, 2, 2, 3, 9], 4, 0, 10)  ->  [3, 1, 0, 1]      bins: 0–2.5, 2.5–5, 5–7.5, 7.5–10
histogram_counts([0, 10, 10], 2, 0, 10)      ->  [1, 2]            10 is the right edge: last bin
\`\`\``,
        starter: `
def histogram_counts(values, bins, lo, hi):
    # TODO: return a list with one count per bin
    pass
`,
        solution: `
def histogram_counts(values, bins, lo, hi):
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        if v < lo or v > hi:
            continue                  # outside the range: ignored
        i = int((v - lo) / width)
        if i >= bins:                 # the right edge belongs to the last bin
            i = bins - 1
        counts[i] += 1
    return counts
`,
        hints: ['The bin of a value v is int((v − lo) / width).', 'For v == hi that formula gives bins, one past the end: clamp it to the last bin.', 'Skip values outside [lo, hi] before computing the bin.'],
        explain: `Every histogram, including \`plt.hist\` and \`DataFrame.plot.hist\`, starts with exactly this counting step. The number of bins changes the picture a lot: too few hides structure, too many turns it into noise, so try several.`,
        tests: [
          { call: 'histogram_counts([1, 2, 2, 3, 9], 4, 0, 10)', expect: '[3, 1, 0, 1]' },
          { name: 'the right edge goes in the last bin', call: 'histogram_counts([0, 10, 10], 2, 0, 10)', expect: '[1, 2]' },
          { name: 'values outside the range are ignored', call: 'histogram_counts([-1, 5, 11], 2, 0, 10)', expect: '[0, 1]' },
          { name: 'random values vs reference', setup: 'import random\nrandom.seed(4)\nvals = [random.uniform(-5, 25) for _ in range(500)]', call: 'histogram_counts(vals, 7, 0, 20)', hidden: true },
          { name: 'matches np.histogram', setup: 'import numpy as np\nrng = np.random.default_rng(2)\nvals = rng.normal(5, 2, 300).round(3).tolist()', call: 'histogram_counts(vals, 6, 0, 12) == np.histogram([v for v in vals if 0 <= v <= 12], bins=6, range=(0, 12))[0].tolist()', expect: 'True', hidden: true }
        ]
      },
      {
        id: 'row-percentages', title: 'Percentages for a Stacked Bar Chart', difficulty: 'easy',
        prompt: `A **100% stacked bar chart** shows, for each group, how its total splits into categories. The data it needs is a table of counts turned into row percentages. Implement \`row_percentages(counts)\`, where \`counts\` is a pandas DataFrame (one row per group, one column per category), returning a new DataFrame of the same shape in which every row is divided by its row total and multiplied by 100. A row whose total is 0 must stay all zeros.

\`\`\`text
         yes   no                 yes    no
men       30   10      ->   men   75.0  25.0
women      5   15           women 25.0  75.0
\`\`\`

The result is ready for \`row_percentages(counts).plot.barh(stacked=True)\`.`,
        starter: `
import pandas as pd

def row_percentages(counts):
    # TODO
    pass
`,
        solution: `
import pandas as pd

def row_percentages(counts):
    totals = counts.sum(axis=1)
    safe = totals.where(totals != 0, 1)          # avoid dividing by zero
    return counts.div(safe, axis=0) * 100
`,
        hints: ['`counts.sum(axis=1)` gives one total per row.', '`counts.div(totals, axis=0)` divides every row by its own total.', 'Replace zero totals by 1 before dividing; the zero counts then stay zero.'],
        explain: `\`div(..., axis=0)\` aligns the totals with the rows, so every row is scaled by its own total: the same idea as normalizing the rows of a confusion matrix to see error rates instead of counts.`,
        tests: [
          { setup: 'import pandas as pd\ncounts = pd.DataFrame({"yes": [30, 5], "no": [10, 15]}, index=["men", "women"])', call: 'row_percentages(counts).round(2).to_dict("list")', expect: '{"yes": [75.0, 25.0], "no": [25.0, 75.0]}' },
          { name: 'a row of zeros stays zero', setup: 'import pandas as pd\ncounts = pd.DataFrame({"a": [0, 2], "b": [0, 2]})', call: 'row_percentages(counts).round(2).to_dict("list")', expect: '{"a": [0.0, 50.0], "b": [0.0, 50.0]}' },
          { name: 'the input is not modified', setup: 'import pandas as pd\ncounts = pd.DataFrame({"a": [1, 3], "b": [1, 1]})', code: 'before = counts.copy()\nrow_percentages(counts)\nassert counts.equals(before), "row_percentages must not change its input"' },
          { name: 'random table vs reference', setup: 'import pandas as pd\nimport numpy as np\nrng = np.random.default_rng(7)\ncounts = pd.DataFrame(rng.integers(0, 50, (6, 4)), columns=list("abcd"))', call: 'row_percentages(counts).round(4).to_dict("list")', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Wes McKinney — Python for Data Analysis, chapter 9: Plotting and Visualization (free online edition)', url: 'https://wesmckinney.com/book/plotting-and-visualization' },
      { title: 'matplotlib — Quick start guide', url: 'https://matplotlib.org/stable/users/explain/quick_start.html' },
      { title: 'pandas — Chart visualization', url: 'https://pandas.pydata.org/docs/user_guide/visualization.html' },
      { title: 'Claus O. Wilke — Fundamentals of Data Visualization (free online book)', url: 'https://clauswilke.com/dataviz/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
