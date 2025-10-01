from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Load trained model
try:
    model = joblib.load('models/bridge_fee_model.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Error details: {traceback.format_exc()}")
    model = None

BRIDGES = ['Hop', 'Wormhole', 'Stargate', 'LI.FI', 'Celer']
CHAINS = ['Ethereum', 'BSC', 'Polygon', 'Avalanche', 'Arbitrum']
TOKENS = ['ETH', 'USDT', 'USDC', 'DAI', 'MATIC']

@app.route('/predict', methods=['POST'])
def predict_fee():
    try:
        # Get data from React frontend
        data = request.json
        print("📥 Received data:", data)
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        source_chain = data.get('sourceChain')
        destination_chain = data.get('destinationChain')
        token = data.get('token')
        preferred_bridge = data.get('bridge')
        
        print(f"🔍 Parsed values: source={source_chain}, dest={destination_chain}, token={token}, bridge={preferred_bridge}")
        
        # Validate inputs
        if not all([source_chain, destination_chain, token, preferred_bridge]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Generate predictions for all bridges
        predictions = {}
        
        for bridge in BRIDGES:
            features = create_features(source_chain, destination_chain, token, bridge)
            
            if features is not None:
                try:
                    # Convert to numpy array and ensure correct shape
                    features_array = np.array(features).reshape(1, -1)
                    print(f"🔧 Features for {bridge}: {features_array.shape}")
                    
                    # Make prediction
                    predicted_fee = model.predict(features_array)[0]
                    predictions[bridge] = max(0, round(float(predicted_fee), 4))
                    print(f"📊 {bridge} prediction: {predictions[bridge]}")
                    
                except Exception as e:
                    print(f"❌ Prediction error for {bridge}: {e}")
                    print(f"Error details: {traceback.format_exc()}")
                    predictions[bridge] = 0
        
        if not predictions:
            return jsonify({'error': 'Could not generate any predictions'}), 500
            
        # Find best alternative
        other_bridges = {k: v for k, v in predictions.items() if k != preferred_bridge}
        if other_bridges:
            best_alternative = min(other_bridges.items(), key=lambda x: x[1])
        else:
            best_alternative = (preferred_bridge, predictions.get(preferred_bridge, 0))
        
        response = {
            'userBridge': preferred_bridge,
            'userFee': predictions.get(preferred_bridge, 0),
            'otherBridges': [{'name': k, 'fee': v} for k, v in other_bridges.items()],
            'best': {'name': best_alternative[0], 'fee': best_alternative[1]}
        }
        
        print("📤 Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def create_features(source_chain, dest_chain, token, bridge):
    """Create feature vector for model prediction"""
    try:
        # Current timestamp
        current_time = datetime.now()
        
        # Create a feature dictionary with all possible columns
        features = {
            'timestamp': current_time.timestamp(),
            'gas_price_gwei': 25.0,
            'network_congestion': 0.5,
            'bridge_latency_sec': 300.0,
            'amount_usd': 1000.0
        }
        
        # Add one-hot encoded features
        for chain in CHAINS:
            features[f'source_chain_{chain}'] = 1 if chain == source_chain else 0
            features[f'destination_chain_{chain}'] = 1 if chain == dest_chain else 0
            
        for t in TOKENS:
            features[f'token_{t}'] = 1 if t == token else 0
            
        for b in BRIDGES:
            features[f'bridge_{b}'] = 1 if b == bridge else 0
        
        # Define the exact column order expected by your model
        feature_columns = [
            'timestamp', 'gas_price_gwei', 'network_congestion', 
            'bridge_latency_sec', 'amount_usd'
        ]
        
        # Add all one-hot encoded columns in consistent order
        for chain in CHAINS:
            feature_columns.extend([f'source_chain_{chain}', f'destination_chain_{chain}'])
        for t in TOKENS:
            feature_columns.append(f'token_{t}')
        for b in BRIDGES:
            feature_columns.append(f'bridge_{b}')
        
        print(f"🔧 Feature columns count: {len(feature_columns)}")
        
        # Build feature vector in correct order
        feature_vector = [features.get(col, 0) for col in feature_columns]
        
        return feature_vector
        
    except Exception as e:
        print(f"❌ Feature creation error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else None
    })

@app.route('/options', methods=['GET'])
def get_options():
    """Return available options for frontend dropdowns"""
    return jsonify({
        'bridges': BRIDGES,
        'chains': CHAINS,
        'tokens': TOKENS
    })

# Add a test endpoint to check feature creation
@app.route('/test-features', methods=['POST'])
def test_features():
    """Test endpoint to check feature creation without prediction"""
    data = request.json
    source_chain = data.get('sourceChain', 'Ethereum')
    dest_chain = data.get('destinationChain', 'BSC')
    token = data.get('token', 'USDT')
    bridge = data.get('bridge', 'Stargate')
    
    features = create_features(source_chain, dest_chain, token, bridge)
    
    return jsonify({
        'features': features,
        'feature_count': len(features) if features else 0,
        'input': {'sourceChain': source_chain, 'destinationChain': dest_chain, 'token': token, 'bridge': bridge}
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)