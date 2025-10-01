import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Load the trained model
model_path = Path(r"C:\Users\RAJENDRAN\Desktop\LINK-X\BACKEND\models\bridge_fee_model.pkl\bridge_fee_model.pkl")
model = joblib.load(model_path)

# Load dataset for encoding reference
csv_path = Path(r'C:\Users\RAJENDRAN\Desktop\LINK-X\BACKEND\dataset\synthetic_bridge_fee_dataset.csv')
df = pd.read_csv(csv_path)

# Preprocess
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.drop(columns=['timestamp'])
df_encoded = pd.get_dummies(df, columns=['source_chain', 'destination_chain', 'token', 'bridge'])

# Get feature columns (excluding target)
all_columns = df_encoded.drop(columns=['fee_usd']).columns

# ---- Get user input ---- #
source_chain = input("Enter source chain: ").strip()
destination_chain = input("Enter destination chain: ").strip()
token = input("Enter token: ").strip()
user_bridge = input("Enter your preferred bridge: ").strip()

# Available bridges from dataset
available_bridges = df['bridge'].unique()

# Store predictions
results = []

# Predict fee for each bridge
for bridge in available_bridges:
    input_dict = {
        f'source_chain_{source_chain}': 1,
        f'destination_chain_{destination_chain}': 1,
        f'token_{token}': 1,
        f'bridge_{bridge}': 1
    }
    input_data = pd.DataFrame([input_dict])

    # Add missing columns with 0s
    for col in all_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Align column order
    input_data = input_data[all_columns]

    # Predict
    try:
        predicted_fee = model.predict(input_data)[0]
        results.append((bridge, predicted_fee))
    except Exception as e:
        print(f"Error predicting for bridge {bridge}: {e}")

# Show fee for user bridge
user_result = next(((b, f) for b, f in results if b.lower() == user_bridge.lower()), None)

if user_result:
    print(f"\n🔎 Estimated Fee for '{user_bridge}': ${user_result[1]:.4f}")
else:
    print(f"\n⚠️ Warning: '{user_bridge}' not found in dataset.")

# Show other options
other_results = [r for r in results if r[0].lower() != user_bridge.lower()]
other_results.sort(key=lambda x: x[1])

print("\n📊 Fees for Other Bridges:")
for bridge, fee in other_results:
    print(f"{bridge}: ${fee:.4f}")

# Best alternative
if other_results:
    best_alt_bridge, best_alt_fee = other_results[0]
    print(f"\n✅ Best Alternative Bridge: {best_alt_bridge} (${best_alt_fee:.4f})")
