from flask_classful import FlaskView
from flask import render_template, request, make_response
from app.utils.file_open import open_model, read_file
from app.routes.engagement_prediction import EngagementPrediction, insert_data_called
import ast
import psycopg2
import config
import pandas as pd

class WithoutScorePrediction(FlaskView):
    def update_csv(self,input_values):
        engagement_prediction_instance = EngagementPrediction()
        new_row_pred = engagement_prediction_instance.fetch_new_row_pred()
        pred_eng = engagement_prediction_instance.use_pred_val()
        new_row = {}
        if "Engagement_Level" not in input_values:
            input_values["Engagement_Level"] = pred_eng
        if not bool(input_values):
            combined_row = {**new_row_pred}
        else:
            print("i am here")
            for feature, value in input_values.items():
                new_row[feature] = [value]
            combined_row = {**new_row_pred, **new_row}
        new_df = pd.DataFrame(combined_row)
        return new_df
    
    def insert_data_to_database(self,new_df,prediction):
        global insert_data_called
        last_row_index = new_df.index[-1]
        target_column = read_file("target_column.txt", "r")
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
            value_tuple += (float(value),)
        if insert_data_called == False:
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
                insert_data_called = True
            except Exception as e:
                print(f"Error inserting data into PostgreSQL table: {str(e)}")
                return "Error inserting data into PostgreSQL table", 500
        return "Successfully updated database"
    
    def post(self):
        try:
            model_pkl = open_model('linear_model_without_score.pkl', 'rb')
        except FileNotFoundError:
            return {'error': 'Model file not found, Please train the model!'}

        selected_features = request.form.get('features_without_score').split(',')
        previous_features = (request.form.get('pred_value'))
        previous_features_list =ast.literal_eval(previous_features)

        if selected_features is None:
            return make_response({"error": "No input data provided in the request body"}, 400)
        
        input_values = {}
        for feature in selected_features:
            if feature in request.form:
                input_values[feature] = request.form[feature]

        try:
            if not bool(input_values):
                final_features = previous_features_list
            else:
                print("previous", previous_features_list)
                input_data = [float(input_values[feature])
                          for feature in selected_features]
                final_features = previous_features_list + input_data
            prediction = model_pkl.predict([final_features])
            if prediction[0] < 0:
                prediction[0] = 0
            new_df = self.update_csv(input_values)
            self.insert_data_to_database(new_df,prediction)
            return render_template('final_result.html')
        except Exception as e:
                return f"{str(e)}", 500

