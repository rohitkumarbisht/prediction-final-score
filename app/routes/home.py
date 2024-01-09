from flask import redirect, render_template, url_for
from flask_classful import FlaskView

from app.utils.file_open import check_file_exists


class Home(FlaskView):
    def get(self):
            return redirect('/swagger')
