import ast

import pandas as pd
import psycopg2
from flask import make_response, request
from flask_classful import FlaskView

import config
from app.routes.engagement_prediction import EngagementPrediction
from app.utils.file_open import open_model


class WithScorePrediction(FlaskView):
    def update_csv(self, input_values):
        engagement_prediction_instance = EngagementPrediction()
        new_row_pred = engagement_prediction_instance.fetch_new_row_pred()
        pred_eng = engagement_prediction_instance.use_pred_val()
        r2_score = self.fetch_r2_score_with_score()
        label = 'High' if r2_score is not None and r2_score >= 0.5 else (
            'Medium' if r2_score is not None and 0.25 <= r2_score < 0.5 else 'Low')
        new_row = {}
        if "predicted_engagement_level" not in input_values:
            input_values["predicted_engagement_level"] = pred_eng
            input_values["prediction_confidence"] = label
        for feature, value in input_values.items():
            new_row[feature] = [value]
        combined_row = {**new_row_pred, **new_row}
        new_df = pd.DataFrame(combined_row)
        return new_df, label

    def insert_data_to_database(self, new_df, prediction):
        last_row_index = new_df.index[-1]
        target_column = ['predicted_final_exam_score']
        # # Update the last row's last column with the predicted value
        new_df.at[last_row_index, target_column[0]] = prediction
        num_columns = new_df.shape[1]
        column_names = new_df.columns.tolist()
        columns_required = ",".join(['"' + col + '"' for col in column_names])
        value_type = (',').join(['%s'] * num_columns)
        insert_query = f"""INSERT INTO {config.db_name}.{config.schema_name}."{config.table_name}"({columns_required}) VALUES ({value_type})"""
        value_tuple = ()
        for i in range(num_columns):
            value = new_df.iloc[-1, i]
            if type(value) is str:
                value_tuple += (value,)
            else:
                value_tuple += (float(value),)
        try:
            conn = psycopg2.connect(
                host=config.db_host,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name
            )
            cursor = conn.cursor()
            cursor.execute(insert_query, value_tuple)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error inserting data into PostgreSQL table: {str(e)}")
            return "Error inserting data into PostgreSQL table", 500
        return "Successfully updated database"

    def fetch_r2_score_with_score(self):
        try:
            conn = psycopg2.connect(
                host=config.db_host,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name
            )
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT r2_score_with_score FROM {config.db_name}.{config.schema_name}.{config.model_config_table} ORDER BY id DESC LIMIT 1;")
            r2_score_with_score = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return float(r2_score_with_score) if r2_score_with_score else None
        except Exception as e:
            print(f"Error fetching r2_score_with_score: {str(e)}")
            return None

    def post(self):
        try:
            model_pkl = open_model('linear_model_with_score.pkl', 'rb')
        except FileNotFoundError:
            return {'error': 'Model file not found, Please train the model!'}

        selected_features = request.form.get('features_with_score').split(',')
        previous_features = (request.form.get('pred_value'))
        previous_features_list = ast.literal_eval(previous_features)

        if selected_features is None:
            return make_response({"error": "No input data provided in the request body"}, 400)
        input_values = {}

        for feature in selected_features:
            if feature in request.form:
                input_values[feature] = float(request.form[feature])
        try:
            new_df, label = self.update_csv(input_values)
            input_data = [float(input_values[feature])
                          for feature in selected_features]
            final_features = previous_features_list + input_data
            prediction = model_pkl.predict([final_features])
            if prediction[0] < 0:
                prediction[0] = 0
            message = self.insert_data_to_database(new_df, prediction)
            # return redirect(url_for('StudentReport:get'))
            return make_response({"message": message, "predicted_score": prediction[0][0], "report_url": "https://app.powerbi.com/reportEmbed?reportId=e005ea73-4088-4798-a173-62bd1a614d92&autoAuth=true&ctid=d7bdcc08-a599-45fe-bbec-f852ee1fd486&navContentPaneEnabled=false", "prediction_confidence": label}, 200)
        except Exception as e:
            return f"{str(e)}", 500
