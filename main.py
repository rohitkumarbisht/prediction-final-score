from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

model = None
model_trained = False

def perform_correlation_analysis(df, target_column):
    # Exclude the target column from the features
    features = [col for col in df.columns if col != target_column]

    correlation_with_G3 = df[features].corrwith(df[target_column])

    correlation_more_than_0_2 = [
        col for col, corr in correlation_with_G3.items()
        if 0.3 < corr
    ]

    correlation_less_than_minus_0_2 = [
        col for col, corr in correlation_with_G3.items()
        if corr < -0.3
    ]

    correlation_minus_0_2_to_0_2 = [
        col for col, corr in correlation_with_G3.items()
        if -0.3 <= corr <= 0.3
    ]

    return correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2


@app.route('/', methods=['GET', 'POST'])
def index():
    global model, model_trained
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            uploaded_file.save('uploaded.csv')
            model_trained = False

            df = pd.read_csv('uploaded.csv')
            target_column = df.columns[-1]
            correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2 = perform_correlation_analysis(df, target_column)

            selected_features = correlation_more_than_0_2 + correlation_less_than_minus_0_2 # Use highly correlated features for the form

            model = GradientBoostingClassifier()
            model.fit(df[selected_features], (df[target_column] >= 10).astype(int))

            model_trained = True

            return render_template('index.html', file_uploaded=True, model_trained=model_trained,
                                   selected_features=selected_features , corr_more_than_0_2=correlation_more_than_0_2,
                                   corr_less_than_minus_0_2=correlation_less_than_minus_0_2,
                                   corr_minus_0_2_to_0_2=correlation_minus_0_2_to_0_2)
    return render_template('index.html', file_uploaded=False, model_trained=model_trained)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_trained:
        return render_template('index.html', file_uploaded=True, model_trained=False)

    # Get form inputs and prepare input data
    selected_features = request.form.getlist('features')
    input_data = []
    for feature in selected_features:
        input_data.append(int(request.form[feature]))

    # Make prediction
    prediction = model.predict([input_data])
    result = "Pass" if prediction[0] == 1 else "Fail"

    return render_template('index.html', prediction_result=result, file_uploaded=True, model_trained=model_trained, selected_features=selected_features)

if __name__ == '__main__':
    app.run(debug=True)
