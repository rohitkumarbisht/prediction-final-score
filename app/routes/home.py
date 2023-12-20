from flask import render_template
from flask_classful import FlaskView


class Home(FlaskView):
    def get(self):
        return render_template("index.html")
