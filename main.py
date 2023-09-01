from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

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
        uploaded_file = request.files['csv_file']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            uploaded_file.save('uploaded.csv')
            model_trained = False

            df = pd.read_csv('uploaded.csv')
            target_column = df.columns[-1]
            correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2 = perform_correlation_analysis(df, target_column)

            selected_features = correlation_more_than_0_2 + correlation_less_than_minus_0_2 # Use highly correlated features for the form

            if selected_features:
                model = LinearRegression()
                model.fit(df[selected_features], df[target_column])

                model_trained = True
            return render_template('index.html', file_uploaded=True, model_trained=model_trained,
                                   selected_features=selected_features , corr_more_than_0_2=correlation_more_than_0_2,
                                   corr_less_than_minus_0_2=correlation_less_than_minus_0_2,
                                   corr_minus_0_2_to_0_2=correlation_minus_0_2_to_0_2)
    return render_template('index.html', file_uploaded=False, model_trained=model_trained)

@app.route('/predict', methods=['POST'])
def predict_and_update_csv():
    if not model_trained:
        return render_template('index.html', file_uploaded=True, model_trained=False)

    # Get form inputs and prepare input data
    selected_features = request.form.get('features').split(',')
    input_values = {}
    for feature in selected_features:
        input_values[feature] = request.form[feature]  # Capture input values

    try:
        # Load the existing CSV
        df = pd.read_csv('uploaded.csv')

        # Create a new row with the input values
        new_row = {}
        for feature, value in input_values.items():
            new_row[feature] = [float(value)]  # Convert to float for consistency

        # Make prediction
        input_data = [float(input_values[feature]) for feature in selected_features]
        prediction = model.predict([input_data])[0]
        
        # Create a DataFrame from the new row
        new_df = pd.DataFrame(new_row)

        last_row_index = new_df.index[-1]
        # Update the last row's last column with the predicted value
        new_df.at[last_row_index, df.columns[-1]] = prediction

        # Concatenate the new row with the existing DataFrame
        updated_df = pd.concat([df, new_df], ignore_index=True)

        # Save the updated DataFrame to the CSV file
        updated_df.to_csv('uploaded.csv', index=False)

        return render_template('index.html', prediction_result=prediction, file_uploaded=True, model_trained=model_trained, selected_features=selected_features)

    except Exception as e:
        print('Error updating CSV or making prediction:', str(e))
        return "Error updating CSV or making prediction", 500

@app.route('/download', methods=['GET'])
def download_csv():
    if os.path.exists('uploaded.csv'):
        return send_file('uploaded.csv', as_attachment=True)
    else:
        return "CSV file not found", 404

if __name__ == '__main__':
    app.run(debug=True)
