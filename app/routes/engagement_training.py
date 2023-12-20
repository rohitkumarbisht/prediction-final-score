from flask_classful import FlaskView
from flask import render_template, make_response, Response
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import contextlib
import time
from datetime import date
from app.routes.distribution_graph import DistributionGraph
from app.utils.file_open import read_file, open_model, save_file, check_file_exists

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

    def save_training_results_to_text(self, accuracy, precision, training_time, date_modified):
        try:
            parameters = {
                'accuracy': accuracy,
                'precision': precision,
                'training_time': training_time,
                'modified_on': date_modified
                }
            data_to_save = "\n".join([f"{key}: {value}" for key, value in parameters.items()])
            save_file(data_to_save,
                              "training_results_eng_level.txt", "w")

        except Exception as e:
            return make_response({"error": f"Failed to save training results to the textfile: {e}"}, 500)

    def get(self):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        actual_columns = read_file("highly_correlated_columns_with_eng_level.txt", "r")
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
        open_model('xgb_model_engagement.pkl','wb',XGB_Model)
        predict = check_file_exists()
        # save training results to text file
        result = self.save_training_results_to_text(
            accuracy, precision, training_time, modified_on)
        if result:
            return result

        return render_template ("training.html", predict=predict ,accuracy= accuracy, precision = precision, training_time = training_time, date_modified = modified_on)