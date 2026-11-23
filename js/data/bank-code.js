/* ML Atlas – extra quiz questions for the Python & Data lessons. They appear only in custom quizzes (#/quiz), never in a
   lesson's own quiz or the review queue. Keyed by lesson id; the format is the lesson quiz format (docs/CONTENT_GUIDE.md). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'python-essentials': [
      { type: 'mcq', q: "What does `[1, 2, 3, 4, 5][1:3]` return?",
        options: ["`[2, 3]`", "`[1, 2, 3]`", "`[2, 3, 4]`", "`[1, 2]`"], answer: 0,
        explain: "A slice starts at its first index (1, the second element) and stops before its second index (3)." },
      { type: 'mcq', q: "What does `{x % 3 for x in range(10)}` produce?",
        options: ["`{0, 1, 2}`", "`[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]`", "`{0, 3, 6, 9}`", "`{1, 2}`"], answer: 0,
        explain: "Curly braces without key: value pairs make a set comprehension, and a set keeps each remainder only once." },
      { type: 'fill', q: "What does `sorted([3, 1, 2], reverse=True)` return?", answer: ["[3, 2, 1]", "[3,2,1]"],
        explain: "`sorted` returns a new list, here in descending order because of `reverse=True`. The original list is left unchanged." },
      { type: 'tf', q: "Tuples can be dictionary keys, but lists cannot.", answer: true,
        explain: "Keys must be hashable. A tuple of hashable items cannot change, so it can be hashed; a list can change, so it cannot." },
      { type: 'mcq', q: "What does `list(zip(['a', 'b'], [1, 2]))` return?",
        options: ["`[('a', 1), ('b', 2)]`", "`['a', 1, 'b', 2]`", "`{'a': 1, 'b': 2}`", "`[('a', 'b'), (1, 2)]`"], answer: 0,
        explain: "`zip` pairs up items by position. Wrapping it in `dict(...)` instead of `list(...)` would give the dictionary." },
      { type: 'mcq', q: "What does `print(f'{3.14159:.2f}')` print?",
        options: ["3.14", "3.14159", "3.1", "3"], answer: 0,
        explain: "The format spec `.2f` shows a float rounded to two decimal places." }
    ],
    'numpy': [
      { type: 'mcq', q: "What is the shape of `np.zeros((3, 4)).T`?",
        options: ["(4, 3)", "(3, 4)", "(12,)", "(1, 12)"], answer: 0,
        explain: "Transposing swaps the axes, so 3 rows and 4 columns become 4 rows and 3 columns." },
      { type: 'mcq', q: "What does `a[a % 2 == 0]` return when `a = np.arange(10)`?",
        options: ["`[0 2 4 6 8]`", "`[1 3 5 7 9]`", "`[True False True ...]`", "`5`"], answer: 0,
        explain: "The comparison builds a boolean mask, and indexing with the mask keeps the elements where it is True." },
      { type: 'tf', q: "Indexing with a boolean mask, as in `a[a > 0]`, returns a copy rather than a view.", answer: true,
        explain: "Boolean masks and integer arrays always copy, while basic slices such as `a[1:3]` return views. Assigning through a mask, `a[a > 0] = 0`, still writes into a." },
      { type: 'mcq', q: "What does `np.argmax([[1, 9, 3], [7, 2, 8]], axis=1)` return?",
        options: ["`[1, 2]`", "`[9, 8]`", "`[1, 0, 1]`", "`1`"], answer: 0,
        explain: "With axis=1, argmax looks along each row and returns the column index of its largest value: 9 is at index 1 and 8 at index 2. Axis 0 would give [1, 0, 1]." },
      { type: 'fill', q: "What is the dtype of `np.array([1, 2, 3]) / 2`?", answer: ["float64", "np.float64", "float"],
        explain: "True division always gives floating point, so the result is float64 even though the input held integers. Use `//` for integer division." }
    ],
    'data-preparation': [
      { type: 'mcq', q: "Where should you compute the mean used to fill in missing values?",
        options: ["On the training set only, then reuse it for validation and test data", "On the full dataset before splitting", "On the test set", "Separately on each set"], answer: 0,
        explain: "A statistic computed on all the data carries information about the test set into training. Fit on training data, then apply those numbers everywhere." },
      { type: 'tf', q: "When some patients appear in several rows, the split should keep all of a patient's rows in the same set.", answer: true,
        explain: "Otherwise the model can recognise a patient from training in the test set, which makes the score too optimistic. GroupKFold and GroupShuffleSplit do this split." },
      { type: 'mcq', q: "Which pandas call counts the missing values in each column?",
        options: ["`df.isna().sum()`", "`df.count()`", "`df.dropna()`", "`df.describe()`"], answer: 0,
        explain: "`isna()` marks each missing cell True and `sum()` counts the True values per column. `count()` gives the non-missing counts instead." },
      { type: 'mcq', q: "Exact duplicate rows appear in both the training set and the test set. What is the effect?",
        options: ["The test score overstates performance on genuinely new data", "Training becomes faster", "Nothing changes", "The test score becomes too pessimistic"], answer: 0,
        explain: "The model is tested partly on examples it has already seen. Remove duplicates before splitting." },
      { type: 'multi', q: "3% of a column's values are missing at random. Which are sensible ways to handle it? (select all)",
        options: ["Fill with the training median", "Fill, and add a column that marks which values were missing", "Fill with the test set's mean", "Drop the affected rows, since there are few"], answer: [0, 1, 3],
        explain: "Training statistics, a missing-value indicator or dropping a few rows are all reasonable. Anything computed on the test set leaks." }
    ],
    'feature-engineering': [
      { type: 'fill', q: "Min-max scaling maps a training range of 20 to 80 onto [0, 1]. What does the value 50 become?", answer: "0.5",
        explain: "(50 − 20) / (80 − 20) = 30 / 60 = 0.5." },
      { type: 'mcq', q: "At prediction time a category appears that was never seen in training. What does `OneHotEncoder(handle_unknown='ignore')` output for it?",
        options: ["All zeros in that feature's one-hot columns", "An error", "A new column", "The most common category"], answer: 0,
        explain: "With handle_unknown='ignore' an unseen category matches none of the known columns, so every one of them is 0. The default setting raises an error instead." },
      { type: 'tf', q: "Tree-based models such as random forests usually need standardized features.", answer: false,
        explain: "Trees split on thresholds, which depend only on the order of the values, so rescaling a feature does not change the splits." },
      { type: 'mcq', q: "Why is coding colours as 1, 2, 3 a poor choice for a linear model?",
        options: ["It invents an order and equal spacing that the colours do not have", "It uses too much memory", "It cannot be undone", "Linear models cannot read integers"], answer: 0,
        explain: "A linear model would treat blue (3) as three times red (1). One-hot encoding gives each colour its own weight." },
      { type: 'mcq', q: "A StandardScaler was fitted on training data. A test value lies far above anything seen in training. What does the scaler output?",
        options: ["A large z-score, possibly well above 3", "Exactly 1", "An error", "NaN"], answer: 0,
        explain: "Standardization is a linear formula, (x − mean) / std, with no clipping. Min-max scaling would likewise give a value above 1." }
    ],
    'pandas-eda': [
      { type: 'mcq', q: "What does `df.groupby('city')['price'].mean()` return?",
        options: ["A Series with one mean price per city", "A single number", "The DataFrame sorted by city", "A DataFrame with one row per house"], answer: 0,
        explain: "groupby splits the rows by city, the selection keeps the price column, and mean combines each group into one value." },
      { type: 'mcq', q: "Which call counts how often each distinct value appears in a column?",
        options: ["`df['col'].value_counts()`", "`df['col'].unique()`", "`df['col'].nunique()`", "`df['col'].describe()`"], answer: 0,
        explain: "`value_counts()` gives a count per value, from most to least common. `nunique()` gives only how many distinct values there are." },
      { type: 'tf', q: "`df.dropna()` changes df in place by default.", answer: false,
        explain: "Like most pandas methods it returns a new DataFrame. Assign the result, `df = df.dropna()`, to keep it." },
      { type: 'mcq', q: "What does `df.pivot_table(index='region', columns='year', values='sales', aggfunc='sum')` produce?",
        options: ["One row per region, one column per year, and total sales in each cell", "One row per sale", "A single total", "A long table with columns region, year and sales"], answer: 0,
        explain: "A pivot table groups by the index and the columns together, then fills each cell with the aggregate of the values." },
      { type: 'fill', q: "How many rows does `df.head()` show by default?", answer: "5",
        explain: "`head()` shows the first 5 rows; `df.head(20)` shows 20. `tail()` shows the last rows in the same way." }
    ],
    'data-visualization': [
      { type: 'mcq', q: "Which chart best shows how a quantity changes over time?",
        options: ["A line chart", "A pie chart", "A histogram", "A box plot"], answer: 0,
        explain: "Time goes on the x axis and the line connects consecutive values, so trends and seasonality stand out." },
      { type: 'mcq', q: "Which call saves a matplotlib figure to a PNG file?",
        options: ["`fig.savefig('plot.png')`", "`plt.show('plot.png')`", "`fig.write('plot.png')`", "`plt.export('plot.png')`"], answer: 0,
        explain: "`savefig` picks the format from the file extension; `dpi=` sets the resolution." },
      { type: 'tf', q: "A pie chart is a good way to compare ten categories with similar values.", answer: false,
        explain: "People judge angles and areas poorly, so small differences between many slices are hard to see. A sorted bar chart shows them clearly." },
      { type: 'mcq', q: "Values in a scatter plot range from 1 to 1,000,000, and most are small. What helps most?",
        options: ["A logarithmic axis", "Switching to a pie chart", "Deleting the large values", "A 3D effect"], answer: 0,
        explain: "A log scale gives each factor of 10 the same space, so small and large values are both visible." },
      { type: 'mcq', q: "Why use `fig, ax = plt.subplots()` rather than only calling `plt.plot`?",
        options: ["It gives explicit Figure and Axes objects, which make multi-panel figures and later changes clear", "It is the only way to draw lines", "It makes plots faster to render", "It saves the figure automatically"], answer: 0,
        explain: "The object-oriented API says exactly which axes each call changes, which matters once a figure has several panels." }
    ],
    'sql-data-engineering': [
      { type: 'order', q: "Put these clauses in the order a database logically evaluates them.",
        items: ["FROM", "WHERE", "GROUP BY", "HAVING", "SELECT", "ORDER BY"],
        explain: "Rows are gathered, filtered, grouped, filtered as groups, projected and finally sorted. That is why ORDER BY can use a column alias from SELECT, while standard SQL does not let WHERE use it." },
      { type: 'mcq', q: "What does `COUNT(DISTINCT user_id)` return?",
        options: ["The number of different users", "The number of rows", "The number of non-null user IDs, repeats included", "The largest user ID"], answer: 0,
        explain: "DISTINCT removes repeats before counting, so a user with many rows counts once." },
      { type: 'tf', q: "An inner join of orders with customers keeps orders whose customer_id has no match in customers.", answer: false,
        explain: "An inner join keeps only rows that match in both tables. A LEFT JOIN from orders keeps every order and fills the customer columns with NULL." },
      { type: 'mcq', q: "Joining users to their orders and then summing each user's credit limit gives totals that are far too large. Why?",
        options: ["Each user row is repeated once per order (join fan-out)", "SUM ignores NULL values", "Joins sort the rows", "The database rounds the values"], answer: 0,
        explain: "A one-to-many join copies the one side. Aggregate the many side to one row per user first, or sum before joining." },
      { type: 'mcq', q: "What does a common table expression, `WITH recent AS (...)`, do?",
        options: ["Names a subquery so the rest of the query can use it like a table", "Creates a permanent table", "Makes every query faster", "Locks the table for writing"], answer: 0,
        explain: "CTEs break a long query into named steps that read top to bottom. They exist only for that one query." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
