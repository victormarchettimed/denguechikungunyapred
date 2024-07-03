from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Carregar o modelo
model = pickle.load(open('model/best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Pontuação Calculada: {output}')

if __name__ == "__main__":
    app.run(debug=True)
