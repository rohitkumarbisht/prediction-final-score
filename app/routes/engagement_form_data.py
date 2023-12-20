from flask import render_template
from flask_classful import FlaskView

import config
from app.utils.file_open import read_file


class EngagementPredictionForm(FlaskView):
    def get(self):
        actual_columns = read_file(
            "highly_correlated_columns_with_eng_level.txt", "r")
        return render_template('engagement_form.html', selected_features=actual_columns, fields=config.fields)
