from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import psycopg2

app = Flask(__name__)

model = None
model_trained = False

# Function to fetch data from the PostgreSQL table
def fetch_data_from_postgresql(database_name, schema_name,table_name):
    global df
    try:
        #connect with the PostgreSQL Server
        conn = psycopg2.connect(
            host='airbyte.cqqg4q5hnscs.ap-south-1.rds.amazonaws.com',
            port='5432',
            user='airbyte',
            password='F648d&lTHltriVidRa0R',
            database=database_name
        )
        cursor = conn.cursor()
        
        # Fetch all data from the table
        cursor.execute(f'SELECT * FROM {database_name}.{schema_name}."STUDENT_DATA"')
        data = cursor.fetchall()
        
        # Get column names from the table
        cursor.execute(f'SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s', (schema_name, table_name,))
        columns = [col[0] for col in cursor.fetchall()]
        
        df = pd.DataFrame(data, columns=columns)
        
        cursor.close()
        conn.close()
        
        return df

    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {str(e)}")
        return None


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
        # Fetch data from the PostgreSQL table
        database_name = 'learninganalytics'
        schema_name = 'learninganalytics'
        table_name = "STUDENT_DATA"
        df = fetch_data_from_postgresql(database_name, schema_name,table_name)

        if df is not None:
            model_trained = False
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
        # Load the db data
        database_name = 'learninganalytics'
        schema_name = 'learninganalytics'
        table_name = "STUDENT_DATA"
        df = fetch_data_from_postgresql(database_name, schema_name,table_name)

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
