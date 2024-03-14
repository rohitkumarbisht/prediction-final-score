import os
import pickle as pkl

import matplotlib.pyplot as plt


def read_file(filename, mode):
    with open(filename, mode) as file:
        return file.read().splitlines()


def save_file(columns, filename, mode):
    if not isinstance(columns, list):
        columns = [columns]
    with open(filename, mode) as file:
        file.write("\n".join(columns))


def open_model(filename, mode, model=None):
    with open(filename, mode) as file:
        if mode == 'wb':
            pkl.dump(model, file)
        elif mode == 'rb':
            return pkl.load(file)
        else:
            return 'mode not defined, please use "rb" or "wb"'


def save_image(image_name):
    try:
        # Ensure the directories exist
        if not os.path.exists("static/images"):
            os.makedirs("static/images")
        image_path = os.path.abspath(f'static/images/{image_name}')
        plt.savefig(image_path, bbox_inches='tight')
    except Exception as e:
        return e
    return image_path


def check_file_exists():
    try:
        xgb_model_engagement_exists = os.path.exists(
            'xgb_model_engagement.pkl')
        reg_model_with_score_exists = os.path.exists(
            'reg_model_with_score.pkl')
        reg_model_without_score_exists = os.path.exists(
            'reg_model_without_score.pkl')
        proceed_prediction = xgb_model_engagement_exists and reg_model_with_score_exists and reg_model_without_score_exists
    except Exception as e:
        return e
    return proceed_prediction
