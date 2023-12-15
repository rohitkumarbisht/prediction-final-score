from flask_classful import FlaskView
from flask import render_template, make_response, Response
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate
import contextlib
import time
import numpy as np
from datetime import date
from app.routes.distribution_graph import DistributionGraph
from app.utils.file_open import read_file, open_model, save_file, check_file_exists

class WithScoreTraining(FlaskView):
    @contextlib.contextmanager
    def open_file(self, filename, mode):
        try:
            with open(filename, mode) as file:
                yield file
        except FileNotFoundError as e:
            return make_response({"error": f"File not found: {e.filename}"}, 404)

    def train_model(self, X, y):
        xgb_model = XGBRegressor()
        xgb_model.fit(X, y)
        return xgb_model
    
    def evaluation_metrics_cal(self,model,X,y):
        scores = cross_validate(model,X,y,cv=10,scoring=('r2','neg_mean_squared_error','neg_mean_absolute_error'), return_train_score=True)
        r2_arr = scores['train_r2']
        mse_arr = -scores['test_neg_mean_squared_error']
        mae_arr = -scores['test_neg_mean_absolute_error']
        r2_score = int(np.mean(r2_arr) * 100) / 100.0
        # np.mean(r2_arr)
        mse = np.mean(mse_arr)
        np.mean(mse_arr)
        mae = np.mean(mae_arr)
        print(mse_arr)
        return r2_score,mse,mae


    def save_training_results_to_text(self, r2, mse, mae, training_time, date_modified):
        try:
            parameters = {
                'r2_score': r2,
                'mean_sqaured_error': mse,
                'mean_absolute_error': mae,
                'training_time': training_time,
                'modified_on': date_modified
                }
            data_to_save = "\n".join([f"{key}: {value}" for key, value in parameters.items()])
            save_file(data_to_save,
                              "training_results_with_score.txt", "w")

        except Exception as e:
            return make_response({"error": f"Failed to save training results to the textfile: {e}"}, 500)

    def get(self):
        csv_data_instance = DistributionGraph()
        csv_data = csv_data_instance.fetch_csv_data()
        actual_columns = read_file("highly_correlated_columns_with_score.txt", "r")
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
        #  Train the Neural Network model
        xgb_model = self.train_model(X, y)
        end_time = time.time()
        # Calculate the training time
        training_time = end_time - start_time
        # calculate accuracy, precision & modified_on
        r2,mse,mae = self.evaluation_metrics_cal(xgb_model,X,y)
        today = date.today()
        modified_on = today.isoformat()
        # Save the model to file
        open_model('xgb_model_with_score.pkl','wb',xgb_model)
        predict = check_file_exists()
        # save training results to text file
        result = self.save_training_results_to_text(
            r2, mse,mae, training_time, modified_on)
        if result:
            return result

        return render_template ("training.html", predict=predict ,r2= r2, mse = mse, mae=mae, training_time = training_time, date_modified = modified_on)