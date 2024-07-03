from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Carregar o modelo treinado e o threshold
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/optimal_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter dados do formulário
    features = []
    for feat in ['fever', 'myalgia', 'headache', 'exanthem', 'vomiting', 'nausea', 'back_pain', 'conjunctivitis', 'arthritis', 'arthralgia', 'petechia', 'retro_orbital_pain']:
        features.append(1 if request.form.get(feat) == 'on' else 0)
    logging.debug(f"Features recebidas: {features}")
    
    # Normalizar os dados
    features_array = np.array(features).reshape(1, -1)
    
    # Fazer a predição
    prediction_prob = model.predict_proba(features_array)[0]
    chikungunya_prob = prediction_prob[0]
    dengue_prob = prediction_prob[1]
    logging.debug(f"Probabilidade de Chikungunya: {chikungunya_prob}")
    logging.debug(f"Probabilidade de Dengue: {dengue_prob}")
    
    prediction = "Chikungunya" if chikungunya_prob >= optimal_threshold else "Dengue"
    logging.debug(f"Predição: {prediction}")
    
    return jsonify({
        'prediction_text': f"Probability of Dengue: {dengue_prob:.2f}, Probability of Chikungunya: {chikungunya_prob:.2f}. Prediction: {prediction}",
        'dengue_prob': dengue_prob,
        'chikungunya_prob': chikungunya_prob,
        'optimal_threshold': optimal_threshold
    })

if __name__ == "__main__":
    app.run(debug=True)
