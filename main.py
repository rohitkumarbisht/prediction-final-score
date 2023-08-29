from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

model = None
model_trained = False
prediction_started = False

def perform_correlation_analysis(df, target_column):
    correlation_with_G3 = df.corr()[target_column]

    correlation_more_than_0_2 = [
        col for col in correlation_with_G3.index
        if 0.14 < correlation_with_G3[col]
    ]

    correlation_less_than_minus_0_2 = [
        col for col in correlation_with_G3.index
        if correlation_with_G3[col] < -0.14
    ]

    correlation_minus_0_2_to_0_2 = [
        col for col in correlation_with_G3.index
        if -0.2 <= correlation_with_G3[col] <= 0.14
    ]

    return correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_trained, prediction_started
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            uploaded_file.save('uploaded.csv')
            model_trained = False
            prediction_started = False

            df = pd.read_csv('uploaded.csv')
            selected_features = ['age', 'failures', 'Medu', 'Fedu', 'G1', 'G2']
            target_column = 'G3'
            df_features = df[selected_features]
            df_target = (df[target_column] >= 10).astype(int)

            correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2 = perform_correlation_analysis(df, target_column)

            return render_template('index.html', file_uploaded=True, model_trained=model_trained,
                                   corr_more_than_0_2=correlation_more_than_0_2,
                                   corr_less_than_minus_0_2=correlation_less_than_minus_0_2,
                                   corr_minus_0_2_to_0_2=correlation_minus_0_2_to_0_2)
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

        correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2 = perform_correlation_analysis(df, target_column)

        model = GradientBoostingClassifier()
        model.fit(df_features, df_target)

        model_trained = True
        return render_template('index.html', file_uploaded=True, model_trained=model_trained,
                               corr_more_than_0_2=correlation_more_than_0_2,
                               corr_less_than_minus_0_2=correlation_less_than_minus_0_2,
                               corr_minus_0_2_to_0_2=correlation_minus_0_2_to_0_2,
                               prediction_started=True)  # Display prediction form after training
    return render_template('index.html', file_uploaded=True, model_trained=model_trained)


@app.route('/predict', methods=['POST'])
def predict():

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
