from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

model = None
model_trained = False
prediction_started = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_trained, prediction_started
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            uploaded_file.save('uploaded.csv')
            model_trained = False
            prediction_started = False
            return render_template('index.html', file_uploaded=True, model_trained=model_trained)
    return render_template('index.html', file_uploaded=False, model_trained=model_trained)

@app.route('/train', methods=['GET', 'POST'])
def train():
    global model, model_trained
    if request.method == 'POST':
        df = pd.read_csv('uploaded.csv')
        selected_features = ['age', 'failures', 'Medu', 'Fedu', 'G1', 'G2']
        target_column = 'G3'
        df_features = df[selected_features]
        df_target = (df[target_column] >= 10).astype(int)
        
        model = GradientBoostingClassifier()
        model.fit(df_features, df_target)
        
        model_trained = True
        return render_template('index.html', file_uploaded=True, model_trained=model_trained)
    return render_template('index.html', file_uploaded=True, model_trained=model_trained)

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    global prediction_started
    prediction_started = True
    return render_template('index.html', file_uploaded=True, model_trained=model_trained, prediction_started=prediction_started)

@app.route('/predict', methods=['POST'])
def predict():
    global model, prediction_started
    if not prediction_started:
        return "Prediction not started yet"
    if model is None:
        return "Model not trained yet"

    # Get form inputs and prepare input data
    age = int(request.form['age'])
    failures = int(request.form['failures'])
    medu = int(request.form['medu'])
    fedu = int(request.form['fedu'])
    g1 = int(request.form['g1'])
    g2 = int(request.form['g2'])
    input_data = [[age, failures, medu, fedu, g1, g2]]
    
    # Make prediction
    prediction = model.predict(input_data)
    result = "Pass" if prediction[0] == 1 else "Fail"
    
    return f"The student is predicted to {result}"

if __name__ == '__main__':
    app.run(debug=True)
