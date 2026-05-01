"""
Run once to produce data/sample_patients.csv (500 synthetic rows).
Usage: python3 data/generate_synthetic.py
"""
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

LOCATIONS = ["femur", "tibia", "radius", "humerus", "fibula", "pelvis", "vertebra"]
GENDERS = ["male", "female"]

N_GOOD = 200
N_MODERATE = 175
N_POOR = 125
N_TOTAL = N_GOOD + N_MODERATE + N_POOR


def _biomarker_series(base, trend_factor, noise_std=5):
    """Generate [day1, week3, week6] for a biomarker with a trend."""
    d1 = base + rng.normal(0, noise_std)
    w3 = d1 * (1 + trend_factor * 0.5) + rng.normal(0, noise_std)
    w6 = d1 * (1 + trend_factor) + rng.normal(0, noise_std)
    return max(d1, 1), max(w3, 1), max(w6, 1)


def _mineral_series(base, noise_std=0.3):
    d1 = base + rng.normal(0, noise_std)
    w3 = base + rng.normal(0, noise_std * 0.8)
    w6 = base + rng.normal(0, noise_std * 0.6)
    return max(d1, 0.1), max(w3, 0.1), max(w6, 0.1)


def generate_group(n, callus_w6_mean, callus_w6_std, bsap_trend, label):
    rows = []
    for i in range(n):
        age = int(rng.integers(20, 80))
        gender = rng.choice(GENDERS)
        loc = rng.choice(LOCATIONS)

        bsap_d1, bsap_w3, bsap_w6 = _biomarker_series(30, bsap_trend, noise_std=6)
        alp_d1, alp_w3, alp_w6 = _biomarker_series(80, bsap_trend * 0.7, noise_std=10)
        p1np_d1, p1np_w3, p1np_w6 = _biomarker_series(50, bsap_trend * 0.9, noise_std=8)

        ca_d1, ca_w3, ca_w6 = _mineral_series(9.5)
        ph_d1, ph_w3, ph_w6 = _mineral_series(3.5)

        cw6 = float(np.clip(rng.normal(callus_w6_mean, callus_w6_std), 10, 400))
        cw3 = float(np.clip(cw6 * rng.uniform(0.4, 0.7), 5, 300))
        cd1 = float(np.clip(cw3 * rng.uniform(0.2, 0.5), 1, 100))

        rows.append({
            "patient_id": f"P{i:04d}_{label}",
            "age": age,
            "gender": gender,
            "fracture_location": loc,
            "bsap_d1": round(bsap_d1, 2),
            "alp_d1": round(alp_d1, 2),
            "p1np_d1": round(p1np_d1, 2),
            "bsap_w3": round(bsap_w3, 2),
            "alp_w3": round(alp_w3, 2),
            "p1np_w3": round(p1np_w3, 2),
            "bsap_w6": round(bsap_w6, 2),
            "alp_w6": round(alp_w6, 2),
            "p1np_w6": round(p1np_w6, 2),
            "ca_d1": round(ca_d1, 2),
            "phos_d1": round(ph_d1, 2),
            "ca_w3": round(ca_w3, 2),
            "phos_w3": round(ph_w3, 2),
            "ca_w6": round(ca_w6, 2),
            "phos_w6": round(ph_w6, 2),
            "callus_d1": round(cd1, 2),
            "callus_w3": round(cw3, 2),
            "callus_w6": round(cw6, 2),
            "healing_category": label,
        })
    return rows


def main():
    rows = (
        generate_group(N_GOOD,     callus_w6_mean=220, callus_w6_std=25, bsap_trend=0.45,  label="Good")
        + generate_group(N_MODERATE, callus_w6_mean=140, callus_w6_std=20, bsap_trend=0.15,  label="Moderate")
        + generate_group(N_POOR,     callus_w6_mean=70,  callus_w6_std=18, bsap_trend=-0.10, label="Poor")
    )

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), "sample_patients.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["healing_category"].value_counts())


if __name__ == "__main__":
    main()
