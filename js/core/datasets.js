/* ML Atlas – datasets that code in the in-browser Python can open as data/<name>, for example
 * pd.read_csv("data/bike_rentals_hourly.csv"). Before code runs, js/engines/python.js writes every dataset the code
 * mentions into Python's file system. The bundled practice datasets live in data/ and are synthetic (made by
 * tools/make-datasets.py, which mimics the shape of the real datasets the sample projects use).
 *   MLA.datasets.list() -> [{ name, title, desc, rows, lesson, bundled }]
 *   MLA.datasets.find(name), MLA.datasets.read(name) -> Promise<Uint8Array>, MLA.datasets.version(name)
 *   MLA.datasets.referenced(code) -> the dataset names a piece of code mentions as data/<name> */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const BUNDLED = [
    { name: 'bike_rentals_hourly.csv', title: 'Bike rentals, hourly (synthetic)', rows: 2184, lesson: 'time-series',
      desc: 'Thirteen weeks of hourly bike rentals with daily and weekly cycles, weather effects, rainy days, two holidays and a slow upward trend. Columns: datetime, hour, weekday (0 is Monday), holiday, temp_c, humidity, rentals.' },
    { name: 'telco_churn_synthetic.csv', title: 'Telecom customer churn (synthetic)', rows: 2000, lesson: 'ensembles',
      desc: 'Two thousand customers of a phone and internet company and whether they left. Columns: customer_id, tenure_months, contract, monthly_charges, total_charges (blank for brand-new customers), internet, support_calls, paperless_billing, churned.' },
    { name: 'card_transactions_synthetic.csv', title: 'Card payments with rare fraud (synthetic)', rows: 5000, lesson: 'anomaly-detection',
      desc: 'Five thousand card payments over 14 days, about 0.9% of them fraudulent. Columns: transaction_id, day, hour, merchant_category, amount, distance_from_home_km, online, new_merchant, fraud.' }
  ].map(d => Object.assign(d, { bundled: true }));

  const cache = {};
  const all = () => BUNDLED.concat(MLA.datasets.uploaded ? MLA.datasets.uploaded() : []);
  function find(name) { return all().find(d => d.name === name) || null; }
  function version(name) { const d = find(name); return d ? (d.bundled ? 'bundled' : String(d.updatedAt || '') + ':' + (d.size || 0)) : ''; }
  function read(name) {
    const d = find(name);
    if (!d) return Promise.reject(new Error('No dataset named ' + name));
    if (!d.bundled) return MLA.datasets.readUploaded(name);
    if (!cache[name]) {
      cache[name] = global.fetch('data/' + name).then(r => { if (!r.ok) throw new Error('Could not download ' + name); return r.arrayBuffer(); })
        .then(b => new Uint8Array(b), e => { delete cache[name]; throw e; });
    }
    return cache[name];
  }
  function referenced(code) {
    const names = new Set(), re = /data\/([A-Za-z0-9_.()-]+)/g;
    let m;
    while ((m = re.exec(String(code || '')))) names.add(m[1]);
    return [...names].filter(n => find(n));
  }

  MLA.datasets = { list: all, find, read, version, referenced };
})(typeof globalThis !== 'undefined' ? globalThis : window);
