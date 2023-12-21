from flask import redirect, render_template, url_for
from flask_classful import FlaskView

from app.utils.file_open import check_file_exists


class Home(FlaskView):
    def get(self):
        if check_file_exists() is True:
            return redirect(url_for('EngagementPredictionForm:get'))
        else:
            return redirect(url_for('DistributionGraph:get'))
        # return render_template("index.html")
