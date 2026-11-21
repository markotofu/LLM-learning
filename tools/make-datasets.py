"""Builds the small practice datasets in data/ (synthetic, seeded, so every run gives the same files).

They let the sample projects start in the in-browser Python before the learner downloads the real dataset. Each one
mimics the shape of a real dataset (the columns, the base rates, the kinds of patterns) but every value is generated
here; js/core/datasets.js describes them to the app. Run: python tools/make-datasets.py
"""
import csv
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(ROOT, exist_ok=True)


def write(name, header, rows):
    with open(os.path.join(ROOT, name), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)
    print(name, len(rows), 'rows')


def bikes():
    """13 weeks of hourly bike rentals with daily, weekly and weather effects and a slow upward trend."""
    rng = np.random.default_rng(7)
    rows = []
    holidays = {4 * 7 + 0, 9 * 7 + 4}                     # two public holidays (day numbers)
    rain = set(rng.choice(91, size=14, replace=False).tolist())
    for day in range(91):
        weekday = day % 7                                  # 0 = Monday
        holiday = int(day in holidays)
        off = weekday >= 5 or holiday
        base_temp = 9 + 12 * day / 90                       # spring warming
        for hour in range(24):
            temp = base_temp + 5 * math.sin((hour - 9) / 24 * 2 * math.pi) + rng.normal(0, 1.2)
            humidity = min(100, max(20, 65 - 1.2 * (temp - 12) + (20 if day in rain else 0) + rng.normal(0, 6)))
            if off:
                shape = 0.15 + 0.85 * math.exp(-((hour - 14) / 3.5) ** 2)
            else:
                shape = 0.08 + math.exp(-((hour - 8) / 1.2) ** 2) + 0.9 * math.exp(-((hour - 17.5) / 1.5) ** 2) + 0.25 * math.exp(-((hour - 12.5) / 2) ** 2)
            level = (60 + 0.35 * day) * shape * (1 + 0.025 * (temp - 15)) * (0.45 if day in rain else 1.0)
            count = int(rng.poisson(max(level, 0.5)))
            stamp = np.datetime64('2024-03-04') + np.timedelta64(day, 'D')
            rows.append([f'{stamp} {hour:02d}:00', hour, weekday, holiday, round(temp, 1), round(humidity), count])
    write('bike_rentals_hourly.csv', ['datetime', 'hour', 'weekday', 'holiday', 'temp_c', 'humidity', 'rentals'], rows)


def churn():
    """Telecom customers: who leaves depends on the contract, tenure, price, service and support calls."""
    rng = np.random.default_rng(11)
    rows = []
    for i in range(2000):
        contract = rng.choice(['month-to-month', 'one-year', 'two-year'], p=[0.55, 0.24, 0.21])
        tenure = int(rng.integers(0, 13) if contract == 'month-to-month' and rng.random() < 0.5 else rng.integers(0, 73))
        internet = rng.choice(['fiber', 'dsl', 'none'], p=[0.44, 0.34, 0.22])
        monthly = {'fiber': 85, 'dsl': 55, 'none': 25}[internet] + rng.normal(0, 9)
        calls = int(rng.poisson(1.2 + (1.0 if internet == 'fiber' else 0)))
        paperless = rng.choice(['yes', 'no'], p=[0.6, 0.4])
        z = (-1.2 + {'month-to-month': 1.3, 'one-year': -0.6, 'two-year': -1.8}[contract] - 0.035 * tenure
             + 0.012 * (monthly - 60) + 0.35 * calls + (0.4 if paperless == 'yes' else 0))
        churned = int(rng.random() < 1 / (1 + math.exp(-z)))
        total = '' if tenure == 0 else round(monthly * tenure * rng.uniform(0.96, 1.04), 2)   # blank for brand-new customers
        rows.append([f'C{i + 1:04d}', tenure, contract, round(monthly, 2), total, internet, calls, paperless, churned])
    write('telco_churn_synthetic.csv', ['customer_id', 'tenure_months', 'contract', 'monthly_charges', 'total_charges', 'internet', 'support_calls', 'paperless_billing', 'churned'], rows)


def fraud():
    """Card payments over 14 days with a rare fraud class that is larger, more often online, at night and far away."""
    rng = np.random.default_rng(23)
    rows = []
    HOURS = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 7, 6, 6, 6, 7, 7, 6, 5, 4, 3, 2], dtype=float)
    HOURS /= HOURS.sum()
    cats = ['grocery', 'fuel', 'restaurant', 'online_retail', 'travel', 'electronics']
    for i in range(5000):
        is_fraud = rng.random() < 0.009
        day = int(rng.integers(0, 14))
        hour = int(rng.choice(24, p=HOURS)) if not is_fraud else int(rng.choice([0, 1, 2, 3, 4, 22, 23, 13]))
        cat = rng.choice(cats, p=[0.3, 0.15, 0.2, 0.2, 0.05, 0.1]) if not is_fraud else rng.choice(cats, p=[0.05, 0.05, 0.05, 0.45, 0.15, 0.25])
        amount = rng.lognormal(3.2, 0.9) if not is_fraud else rng.lognormal(5.0, 0.8)
        distance = rng.exponential(8) if not is_fraud else rng.exponential(300)
        online = int(cat == 'online_retail' or rng.random() < (0.7 if is_fraud else 0.1))
        new_merchant = int(rng.random() < (0.75 if is_fraud else 0.12))
        rows.append([i + 1, day, hour, cat, round(float(amount), 2), round(float(distance), 1), online, new_merchant, int(is_fraud)])
    write('card_transactions_synthetic.csv', ['transaction_id', 'day', 'hour', 'merchant_category', 'amount', 'distance_from_home_km', 'online', 'new_merchant', 'fraud'], rows)


if __name__ == '__main__':
    bikes()
    churn()
    fraud()
