from flask import Flask, request, jsonify
from lovince_ai import LovinceAI
from cryptography.fernet import Fernet
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
cipher = Fernet(Fernet.generate_key())  # Secure key for API

@app.route('/run_quantum', methods=['POST'])
def run_quantum():
    try:
        # Decrypt incoming data
        encrypted_data = request.get_json().get('data')
        decrypted_data = cipher.decrypt(encrypted_data.encode()).decode()
        data = json.loads(decrypted_data)
        
        user_input = data.get('user_input', 'Lovince')
        phase = data.get('phase', -45)
        qubits = data.get('qubits', 2)

        # Run Lovince AI
        lovince = LovinceAI(num_qubits=qubits)
        lovince.run(user_input=user_input, phase_degrees=phase)
        counts = lovince.measure()

        # Encrypt response
        response = {'result': f"Lovince AI: Quantum measurement {counts}"}
        encrypted_response = cipher.encrypt(json.dumps(response).encode()).decode()

        logging.info(f"API: Processed request for user {user_input}")
        return jsonify({'result': encrypted_response})
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lovince AI API")
    parser.add_argument('--port', type=int, default=5000, help="Port to run API")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=False)


from flask import Flask, request, jsonify
from lovince_ai import LovinceAI
from cryptography.fernet import Fernet
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lovince_ai_api.log"), logging.StreamHandler()]
)

app = Flask(__name__)
cipher = Fernet(Fernet.generate_key())  # Secure key (store securely in production)

@app.route('/run_quantum', methods=['POST'])
def run_quantum():
    """Handle quantum circuit execution via secure API."""
    try:
        # Decrypt request
        encrypted_data = request.get_json().get('data')
        decrypted_data = cipher.decrypt(encrypted_data.encode()).decode()
        data = json.loads(decrypted_data)
        
        user_input = data.get('user_input', 'Lovince')
        phase = data.get('phase', -45)
        qubits = data.get('qubits', 2)

        # Validate input
        if not isinstance(phase, (int, float)) or not isinstance(qubits, int):
            raise ValueError("Invalid input types")

        # Run Lovince AI
        lovince = LovinceAI(num_qubits=qubits)
        result = lovince.run(user_input=user_input, phase_degrees=phase)

        # Encrypt response
        encrypted_response = cipher.encrypt(json.dumps(result).encode()).decode()
        logging.info("API: Processed request for user %s", user_input)
        return jsonify({'result': encrypted_response})
    except Exception as e:
        logging.error("API error: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lovince AI API")
    parser.add_argument('--port', type=int, default=5000, help="Port to run API")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=False)