from flask import render_template
from flask_classful import FlaskView


class StudentReport(FlaskView):
    def get(self):
        return render_template("final_result.html")