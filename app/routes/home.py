from flask import redirect
from flask_classful import FlaskView


class Home(FlaskView):
    def get(self):
            return redirect('/swagger')
