import contextlib
import time
from datetime import date

import psycopg2
from flask import Response, make_response
from flask_classful import FlaskView
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

import config
from app.routes.distribution_graph import DistributionGraph
from app.utils.file_open import (check_file_exists, open_model, read_file,
                                 save_file)


class EngagementTraining(FlaskView):
    @contextlib.contextmanager
    def open_file(self, filename, mode):
        try:
            with open(filename, mode) as file:
                yield file
        except FileNotFoundError as e:
            return make_response({"error": f"File not found: {e.filename}"}, 404)

    def train_model(self, X, y):
        XGB_Model = XGBClassifier()
        XGB_Model.fit(X, y)
        return XGB_Model

    def save_training_results_to_database(self, accuracy, precision, training_time, date_modified):
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
                        update_sql = f"UPDATE {config.schema_name}.{config.model_config_table} SET accuracy_eng_level = %s, precision_eng_level = %s, training_time_eng_level = %s, modified_on_eng_level = %s WHERE id = %s;"
                        cursor.execute(
                            update_sql, (accuracy, precision, training_time, date_modified, last_inserted_id))
                        conn.commit()
                    else:
                        sql = f"INSERT INTO {config.schema_name}.{config.model_config_table} (accuracy_eng_level, precision_eng_level, training_time_eng_level, modified_on_eng_level) VALUES (%s, %s, %s, %s);"
                        value_tuple = (accuracy, precision,
                                       training_time, date_modified)
                        cursor.execute(sql, value_tuple)
                        conn.commit()
        except Exception as e:
            return make_response({"error": f"Failed to save training results to the database: {e}"}, 500)

    def save_training_results_to_text(self, accuracy, precision, training_time, date_modified):
        try:
            parameters = {
                'accuracy': accuracy,
                'precision': precision,
                'training_time': training_time,
                'modified_on': date_modified
            }
            data_to_save = "\n".join(
                [f"{key}: {value}" for key, value in parameters.items()])
            save_file(data_to_save,
                      "training_results_eng_level.txt", "w")

        except Exception as e:
            return make_response({"error": f"Failed to save training results to the textfile: {e}"}, 500)

    def get(self):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        actual_columns = read_file(
            "highly_correlated_columns_with_eng_level.txt", "r")
        if isinstance(actual_columns, Response):
            return actual_columns
        selected_column = read_file("target_column_eng_level.txt", "r")
        if isinstance(selected_column, Response):
            return selected_column
        X = csv_data[actual_columns]
        y = csv_data[selected_column]
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Start measuring the training time
        start_time = time.time()
        #  Train the Neural Network model
        XGB_Model = self.train_model(X, y)
        end_time = time.time()
        # Calculate the training time
        training_time = end_time - start_time
        # calculate accuracy, precision & modified_on
        accuracy = cross_val_score(
            XGB_Model, X, y, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(
            XGB_Model, X, y, cv=cv, scoring='precision_macro').mean()
        today = date.today()
        modified_on = today.isoformat()
        # Save the model to file
        open_model('xgb_model_engagement.pkl', 'wb', XGB_Model)
        predict = check_file_exists()
        # save training results to text file
        self.save_training_results_to_database(
            accuracy, precision, training_time, modified_on)
        result = self.save_training_results_to_text(
            accuracy, precision, training_time, modified_on)
        if result:
            return result

        # return render_template("training.html", predict=predict, accuracy=accuracy, precision=precision, training_time=training_time, date_modified=modified_on)
        return make_response({"message": "model trained sucessfully", "accuracy": accuracy, "precision": precision, "date_modified": modified_on, "training_time": training_time}, 200)
