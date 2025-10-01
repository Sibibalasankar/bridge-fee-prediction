import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Load dataset
csv_path = Path(r'C:\Users\RAJENDRAN\Desktop\LINK-X\BACKEND\dataset\synthetic_bridge_fee_dataset.csv')

df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.drop(columns=['timestamp'])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['source_chain', 'destination_chain', 'token', 'bridge'])

# Split into features and target
X = df_encoded.drop(columns=['fee_usd'])
y = df_encoded['fee_usd']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Ensure models folder exists
model_dir = Path(r'C:\Users\RAJENDRAN\Desktop\LINK-X\BACKEND\models\bridge_fee_model.pkl')

model_dir.mkdir(parents=True, exist_ok=True)

# Save the model
model_path = model_dir / 'bridge_fee_model.pkl'
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at: {model_path}")
