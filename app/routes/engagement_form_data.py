from flask import redirect, render_template, url_for
from flask_classful import FlaskView

import config
from app.utils.file_open import check_file_exists, read_file


class EngagementPredictionForm(FlaskView):
    def get(self):
        if check_file_exists() is False:
            return redirect(url_for('DistributionGraph:get'))
        else:
            actual_columns = read_file(
                "highly_correlated_columns_with_eng_level.txt", "r")
            return render_template('engagement_form.html', selected_features=actual_columns, fields=config.fields)
