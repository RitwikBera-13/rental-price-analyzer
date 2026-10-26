import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CSV_PATH = "data/mumbai_rentals.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "data/sample_rent_data.csv"

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"Missing training dataset: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Features matching the active ML pipeline
if "zone" in df.columns:
    cat_features = ["zone"]
    num_features = ["area_sqft", "bedrooms", "bathrooms"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    X = df[cat_features + num_features]
else:
    num_features = ["area_sqft", "bedrooms", "bathrooms"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[num_features]

y = df["rent"]
model.fit(X, y)

os.makedirs("models", exist_ok=True)
with open("models/rent_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to models/rent_model.pkl")
