from flask_classful import FlaskView
from flask import render_template

class Home(FlaskView):
    def get(self):
        return render_template("index.html")