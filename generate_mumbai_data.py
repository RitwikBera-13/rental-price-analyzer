import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 5000

# Mumbai rental price patterns (approx psf)
zones = ["South Mumbai", "Central Mumbai", "Western Suburbs", "Eastern Suburbs", "Navi Mumbai"]
zone_mean_rent_psf = {
    "South Mumbai": 150,
    "Central Mumbai": 100,
    "Western Suburbs": 80,
    "Eastern Suburbs": 70,
    "Navi Mumbai": 50
}

data = []
for _ in range(n):
    zone = np.random.choice(zones, p=[0.1, 0.2, 0.3, 0.2, 0.2])
    area_sqft = np.random.randint(300, 2000)
    bedrooms = max(1, int(np.round(area_sqft / 400)))
    bathrooms = max(1, bedrooms - np.random.randint(0, 2))
    furnished = np.random.choice([0,1], p=[0.7,0.3])
    age = np.random.randint(0, 40)

    base_psf = zone_mean_rent_psf[zone]

    rent = area_sqft * base_psf
    rent *= (1 + furnished * 0.10)
    rent *= np.random.uniform(0.85, 1.20)
    rent = round(rent, 2)

    data.append({
        "zone": zone,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "furnished": furnished,
        "age_years": age,
        "rent": rent
    })

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
out_path = "data/mumbai_rentals.csv"

df.to_csv(out_path, index=False)
print(f"Saved synthetic Mumbai dataset to {out_path}")
