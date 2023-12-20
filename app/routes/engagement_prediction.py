from flask import request, make_response, render_template
from flask_classful import FlaskView
from app.utils.file_open import read_file, open_model
import pandas as pd
import numpy as np
import json
import psycopg2
import config

insert_data_called = False

class EngagementPrediction(FlaskView):
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EngagementPrediction, cls).__new__(cls)
            cls._instance.new_row_pred = None
            cls._instance.eng_pred = None
        return cls._instance
    
    def fetch_new_row_pred(self):
        if self.new_row_pred is None:
            return None
        else:
            return self.new_row_pred
    
    def use_pred_val(self):
        return self.eng_pred   
     
    def get_final_score_columns(self):
        engagement_columns = read_file("highly_correlated_columns_with_eng_level.txt", "r")
        without_score_column = read_file("highly_correlated_columns_without_score.txt", "r")
        with_score_column = read_file("highly_correlated_columns_with_score.txt", "r")
        without_score = [item for item in without_score_column if item not in engagement_columns]
        with_score = [item for item in with_score_column if item not in engagement_columns]
        return with_score, without_score

    def post(self):
        # Load the model
        try:
            model_pkl = open_model('xgb_model_engagement.pkl', 'rb')
        except FileNotFoundError:
            return make_response({'error': 'Model file not found, Please train the model!'}, 404)

        selected_features = request.form.get('features').split(',')
        if selected_features is None:
            return make_response({"error": "No input data provided in the request body"}, 400)
        
        input_values = {}
        for feature in selected_features:
            input_values[feature] = request.form[feature]  # Capture input values

        try:
            # Create a new row with the input values
            new_row_pred = {}
            for feature, value in input_values.items():
                new_row_pred[feature] = [value]
            
            self.new_row_pred = new_row_pred
            # Make prediction
            input_data = [float(input_values[feature])
                        for feature in selected_features]
            pred_engagement = model_pkl.predict([input_data])
            self.eng_pred = pred_engagement[0]
            with_score, without_score = self.get_final_score_columns()

        except Exception as e:
            return f"{str(e)}", 500

        # Predict using the model
        if len(pred_engagement) == 0:
            return make_response({"error": "Prediction failed as data can't be processed"},422)

        return render_template('student_final_score_pred.html', engagement=pred_engagement[0],with_score=with_score, without_score=without_score ,fields=config.fields, pred_value=[input_data] )

