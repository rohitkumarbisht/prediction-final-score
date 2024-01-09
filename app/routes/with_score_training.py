import contextlib
import time
from datetime import date

import numpy as np
import psycopg2
from flask import Response, make_response
from flask_classful import FlaskView
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

import config
from app.routes.distribution_graph import DistributionGraph
from app.utils.file_open import (check_file_exists, open_model, read_file,
                                 save_file)


class WithScoreTraining(FlaskView):
    @contextlib.contextmanager
    def open_file(self, filename, mode):
        try:
            with open(filename, mode) as file:
                yield file
        except FileNotFoundError as e:
            return make_response({"error": f"File not found: {e.filename}"}, 404)

    def train_model(self, X, y):
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        return linear_model

    def evaluation_metrics_cal(self, model, X, y):
        scores = cross_validate(model, X, y, cv=10, scoring=(
            'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), return_train_score=True)
        r2_arr = scores['test_r2']
        mse_arr = -scores['test_neg_mean_squared_error']
        mae_arr = -scores['test_neg_mean_absolute_error']
        r2_score = int((r2_arr[0]) * 100) / 100.0
        mse = np.mean(mse_arr)
        mae = np.mean(mae_arr)
        return r2_score, mse, mae

    def save_training_results_to_database(self, r2, training_time, date_modified):
        try:
            with psycopg2.connect(
                dbname=config.db_name, user=config.db_user, password=config.db_password, host=config.db_host, port=config.db_port
            ) as conn:
                with conn.cursor() as cursor:
                    # Step 1: Retrieve the values from the previous row
                    select_previous_row_sql = f"SELECT * FROM {config.schema_name}.{config.model_config_table} ORDER BY id DESC LIMIT 1;"
                    cursor.execute(select_previous_row_sql)
                    previous_row = cursor.fetchone()

                    if previous_row:
                        # Step 2: Create a copy of the values from the previous row
                        previous_values = list(previous_row)

                        # Step 3: Insert the copy of those values into a new row
                        column_names = [col.name for col in cursor.description]
                        insert_sql = f"INSERT INTO {config.schema_name}.{config.model_config_table} ({', '.join(column_names[1:])}) VALUES ({', '.join(['%s'] * (len(column_names) - 1))}) RETURNING *;"
                        cursor.execute(insert_sql, previous_values[1:])
                        last_inserted_id = cursor.fetchone()[0]

                        # Step 4: Update specific columns with the new values in the new row using UPDATE command
                        update_sql = f"UPDATE {config.schema_name}.{config.model_config_table} SET r2_score_with_score = %s, training_time_with_score = %s, modified_on_with_score = %s WHERE id = %s;"
                        cursor.execute(
                            update_sql, (r2, training_time, date_modified, last_inserted_id))
                        conn.commit()
                    else:
                        sql = f"INSERT INTO {config.schema_name}.{config.model_config_table} (r2_score_with_score, training_time_with_score, modified_on_with_score) VALUES (%s, %s, %s, %s);"
                        value_tuple = (r2, training_time, date_modified)
                        cursor.execute(sql, value_tuple)
                        conn.commit()
        except Exception as e:
            return make_response({"error": f"Failed to save training results to the database: {e}"}, 500)

    def save_training_results_to_text(self, r2, mse, mae, training_time, date_modified):
        try:
            parameters = {
                'r2_score': r2,
                'mean_sqaured_error': mse,
                'mean_absolute_error': mae,
                'training_time': training_time,
                'modified_on': date_modified
            }
            data_to_save = "\n".join(
                [f"{key}: {value}" for key, value in parameters.items()])
            save_file(data_to_save,
                      "training_results_with_score.txt", "w")

        except Exception as e:
            return make_response({"error": f"Failed to save training results to the textfile: {e}"}, 500)

    def get(self):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        actual_columns = read_file(
            "highly_correlated_columns_with_score.txt", "r")
        if isinstance(actual_columns, Response):
            return actual_columns
        selected_column = read_file("target_column.txt", "r")
        if isinstance(selected_column, Response):
            return selected_column
        X = csv_data[actual_columns]
        y = csv_data[selected_column]
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Start measuring the training time
        start_time = time.time()
        linear_model = self.train_model(X, y)
        end_time = time.time()
        # Calculate the training time
        training_time = end_time - start_time
        # calculate accuracy, precision & modified_on
        r2, mse, mae = self.evaluation_metrics_cal(linear_model, X, y)
        today = date.today()
        modified_on = today.isoformat()
        # Save the model to file
        open_model('linear_model_with_score.pkl', 'wb', linear_model)
        predict = check_file_exists()
        self.save_training_results_to_database(r2, training_time, modified_on)
        result = self.save_training_results_to_text(
            r2, mse, mae, training_time, modified_on)
        if result:
            return result

        # return render_template("training.html", predict=predict, r2=r2, mse=mse, mae=mae, training_time=training_time, date_modified=modified_on)
        return make_response({"message": "model trained sucessfully", "r2_score": r2, "mean_absolute_error": mae, "mean_squared_error": mse, "training_time": training_time, "date_modified": modified_on}, 200)
