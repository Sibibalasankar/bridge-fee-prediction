# 🔗 Bridge Fee Predictor (Backend)

This project is a **Machine Learning-powered backend** for predicting **cross-chain bridge fees**.  
It uses a synthetic dataset of blockchain transactions to train a model that estimates transfer fees across different bridges.  

---

## 📂 Project Structure

```
BACKEND/
│
├── dataset/
│   └── synthetic_bridge_fee_dataset.csv   # Dataset of synthetic bridge transactions
│
├── models/
│   └── bridge_fee_model.pkl               # Trained ML model (saved here)
│
├── train/
│   ├── train_fee_model.py                 # Script to train the model
│   └── evaluate_model.py                  # Script to test/predict with the model
```

---

## 📊 Dataset

The dataset (`synthetic_bridge_fee_dataset.csv`) contains transaction records with the following columns:

| Column              | Description                                                   |
|---------------------|---------------------------------------------------------------|
| timestamp           | Time of the transaction                                       |
| source_chain        | Blockchain where transfer starts (e.g., Ethereum, BSC)        |
| destination_chain   | Blockchain where transfer ends                                |
| token               | Token being transferred (ETH, USDT, DAI, USDC, etc.)          |
| amount_usd          | Transfer amount in USD                                        |
| gas_price_gwei      | Gas price at that moment                                      |
| network_congestion  | Congestion level of the network (0–1 scale)                   |
| bridge_latency_sec  | Time taken for transfer (in seconds)                          |
| bridge              | Name of the bridge used (Hop, Wormhole, Stargate, etc.)       |
| fee_usd             | Transaction fee in USD (this is the target to predict)        |

---

## ⚙️ Model Training

The model is trained using a **Random Forest Regressor**.

1. **Run training script:**
   ```bash
   python train/train_fee_model.py
   ```

2. **What happens inside:**
   - Loads dataset
   - Converts categorical variables (chains, tokens, bridges) into numeric values (one-hot encoding)
   - Splits features (`X`) and target (`fee_usd`)
   - Trains the Random Forest model
   - Saves model as `models/bridge_fee_model.pkl`

---

## 🔮 Model Evaluation (Prediction)

You can use the trained model to predict fees for different bridges.

1. **Run evaluation script:**
   ```bash
   python train/evaluate_model.py
   ```

2. **Provide input when prompted:**
   ```
   Enter source chain: Ethereum
   Enter destination chain: BSC
   Enter token: USDT
   Enter your preferred bridge: Stargate
   ```

3. **Example Output:**
   ```
   🔎 Estimated Fee for 'Stargate': $5.1400

   📊 Fees for Other Bridges:
   Hop: $24.6400
   Wormhole: $29.0600
   LI.FI: $21.4900
   Celer: $24.8300

   ✅ Best Alternative Bridge: LI.FI ($21.4900)
   ```

---

## 🛠️ Requirements

Install dependencies before running scripts:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## 🚀 Future Improvements

- Use **real blockchain data** instead of synthetic
- Add **Flask/FastAPI endpoints** for serving predictions as APIs
- Include **visualizations** of bridge fee trends
- Experiment with **advanced ML models** (XGBoost, LightGBM, Neural Networks)

---

## 👨‍💻 Author

Developed as part of a **Bridge Fee Prediction** project using **Python + Scikit-learn**.
