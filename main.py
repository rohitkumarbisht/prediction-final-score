from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import psycopg2

app = Flask(__name__)

model = None
model_trained = False

# Define your PostgreSQL database connection settings
db_host = 'airbyte.cqqg4q5hnscs.ap-south-1.rds.amazonaws.com'
db_port = 5432
db_user = 'airbyte'
db_password = 'F648d&lTHltriVidRa0R'
db_name = 'learninganalytics'
schema_name = 'learninganalytics'
table_name = "STUDENT_DATA"

# Function to fetch data from the PostgreSQL table

fields = [
    ('Student_ID', 'Student id'), ('No_of_Logins', 'No. of logins'), ('ContentReads',
                                                                      'Content read'), ('ForumReads', 'Forum read'),
    ('ForumPosts', 'Forum posts'), ('Quiz_Reviews_before_submission',
                                    'No. of quiz review before submission'), ('Assignment1_delay', 'Assignment 1 delay'),
    ('Assignment2_delay', 'Assignment 2 delay'), ('Assignment3_delay',
                                                  'Assignment 3 delay'), ('Assignment1_submit', 'Assignment 1 submit time'),
    ('Assignment2_submit', 'Assignment 2 submit time'), ('Assignment3_submit',
                                                         'Assignment 3 submit time'),
    ('Average_time_to_submit_assignment',
     'Average time to submit assignments'), ('Engagement_Level', 'Engagement level'),
    ('assignment1_score', 'Assignment 1 score'), ('assignment2_score',
                                                  'Assignment 2 score'), ('assignment3_score', 'Assignment 3 score'),
    ('quiz1_score', 'Quiz 1 score'), ('Midterm_exam_score',
                                      'Midterm exam score'), ('final_exam_score', 'Final exam score'),
]


def fetch_data_from_postgresql():
    global df
    try:
        # connect with the PostgreSQL Server
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        # Fetch all data from the table
        cursor.execute(f'SELECT * FROM {db_name}.{schema_name}."{table_name}"')
        data = cursor.fetchall()

        # Get column names from the table
        cursor.execute(
            'SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s', (schema_name, table_name,))
        columns = [col[0] for col in cursor.fetchall()]

        df = pd.DataFrame(data, columns=columns)

        cursor.close()
        conn.close()

        return df

    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {str(e)}")
        return None


def corr_with_el(eng_data, eng_target):
    features = [col for col in eng_data.columns if col != eng_target]
    corr_with_eng = eng_data[features].corrwith(eng_data[eng_target])

    corr_col_gt_0_5 = [
        col for col, corr in corr_with_eng.items()
        if corr >= 0.5
    ]

    corr_col_lt_minus_0_5 = [
        col for col, corr in corr_with_eng.items()
        if corr <= -0.5
    ]
    return corr_col_gt_0_5, corr_col_lt_minus_0_5


def perform_correlation_analysis(df, target_column):
    # Exclude the target column from the features
    features = [col for col in df.columns if col != target_column]

    corr_with_target_col = df[features].corrwith(df[target_column])

    correlation_more_than_0_2 = [
        col for col, corr in corr_with_target_col.items()
        if 0.2 < corr
    ]

    correlation_less_than_minus_0_2 = [
        col for col, corr in corr_with_target_col.items()
        if corr < -0.2
    ]

    correlation_minus_0_2_to_0_2 = [
        col for col, corr in corr_with_target_col.items()
        if -0.2 <= corr <= 0.2
    ]

    return correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2


@app.route('/', methods=['GET'])
def index():
    global rf_model, rf_model_trained, data, eng_data, insert_data_called
    if request.method == 'GET':
        insert_data_called = False
        rf_model_trained = False
        # Fetch data from the PostgreSQL table
        data = fetch_data_from_postgresql()
        print(data.columns)
        eng_data = data.drop(columns=[data.columns[0], data.columns[14], data.columns[15],
                             data.columns[16], data.columns[17], data.columns[18], data.columns[19]])
        eng_target = 'Engagement_Level'
        if eng_data is not None:
            corr_col_gt_0_5, corr_col_lt_minus_0_5 = corr_with_el(
                eng_data, eng_target)
            # Use highly correlated features for the form
            selected_features = corr_col_gt_0_5 + corr_col_lt_minus_0_5

            if selected_features:
                rf_model = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                rf_model.fit(data[selected_features], data[eng_target])
                rf_model_trained = True
    return render_template('index.html', selected_features=selected_features, model_trained=rf_model_trained, fields=fields)


@app.route('/final-score', methods=['GET', 'POST'])
def final_score():
    global model, model_trained, df, insert_data_called, pred_engagement, new_row_pred, rf_model, TSS
    selected_features = request.form.get('features').split(',')
    input_values = {}
    for feature in selected_features:
        input_values[feature] = request.form[feature]  # Capture input values

    try:
        # Create a new row with the input values
        new_row_pred = {}
        for feature, value in input_values.items():
            new_row_pred[feature] = [value]
        # Make prediction
        input_data = [float(input_values[feature])
                      for feature in selected_features]
        pred_engagement = rf_model.predict([input_data])

    except Exception as e:
        print('Error updating CSV or making prediction:', str(e))
        return "Error updating CSV or making prediction", 500

    if request.method == 'POST':
        insert_data_called = False
        columns_to_drop = data.columns[0:13]
        df = data.drop(columns=columns_to_drop)
        if df is not None:
            model_trained = False
            target_column = df.columns[-1]
            correlation_more_than_0_2, correlation_less_than_minus_0_2, correlation_minus_0_2_to_0_2 = perform_correlation_analysis(
                df, target_column)

            # Use highly correlated features for the form
            selected_features = correlation_more_than_0_2 + correlation_less_than_minus_0_2

            if selected_features:
                mean_y = np.mean(df['final_exam_score'])
                TSS = np.sum((df['final_exam_score'] - mean_y) ** 2)
                model = LinearRegression()
                model.fit(df[selected_features], df[target_column])

                model_trained = True
    return render_template('student_final_score_pred.html', selected_features=selected_features, engagement=pred_engagement[0], correlation_less_than_minus_0_2=correlation_less_than_minus_0_2, correlation_more_than_0_2=correlation_more_than_0_2, model_trained=model_trained, fields=fields)


@app.route('/predict', methods=['POST'])
def predict_and_update_csv():
    global insert_data_called
    if not model_trained:
        return "Train the model again", 500

    # Get form inputs and prepare input data
    selected_features = request.form.get('features').split(',')
    input_values = {}

    for feature in selected_features:
        # Capture input values if they exist
        if feature in request.form:
            input_values[feature] = request.form[feature]

    # Manually add Engagement_Level if it's not in the form
        if "Engagement_Level" not in input_values:
            input_values["Engagement_Level"] = pred_engagement[0]

    try:
        # Create a new row with the input values
        new_row = {}
        for feature, value in input_values.items():
            new_row[feature] = [value]
        # Make prediction
        input_data = [float(input_values[feature])
                      for feature in selected_features]
        y_pred = model.predict(df[selected_features])
        prediction = model.predict([input_data])[0]
        RSS = np.sum((df['final_exam_score'] - y_pred) ** 2)
        R2 = 1 - (RSS / TSS)  # R2 Score

        if prediction < 0:
            prediction = 0

        combined_row = {**new_row_pred, **new_row}
        # Create a DataFrame from the new row
        new_df = pd.DataFrame(combined_row)

        last_row_index = new_df.index[-1]
        # Update the last row's last column with the predicted value
        new_df.at[last_row_index, data.columns[-1]] = prediction
        num_columns = new_df.shape[1]
        column_names = new_df.columns.tolist()
        columns_required = ",".join(['"' + col + '"' for col in column_names])
        value_type = (',').join(['%s'] * num_columns)

        insert_query = f"""INSERT INTO {db_name}.{schema_name}."{table_name}"({columns_required}) VALUES ({value_type})"""

        value_tuple = ()
        for i in range(num_columns):
            value = new_df.iloc[-1, i]
            value_tuple += (float(value),)

        if insert_data_called == False:
            # Call insert_data() only once
            try:
                conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    password=db_password,
                    database=db_name
                )
                cursor = conn.cursor()
                cursor.execute(insert_query, value_tuple)
                conn.commit()
                cursor.close()
                conn.close()
                insert_data_called = True

            except Exception as e:
                print(f"Error inserting data into PostgreSQL table: {str(e)}")
                return "Error inserting data into PostgreSQL table", 500

        return render_template('final_result.html', prediction_result=insert_data_called, file_uploaded=True, model_trained=model_trained, selected_features=selected_features)

    except Exception as e:
        print('Error updating CSV or making prediction:', str(e))
        return "Error updating CSV or making prediction", 500


if __name__ == '__main__':
    app.run(debug=True)
