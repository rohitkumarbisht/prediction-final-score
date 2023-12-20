from flask import Flask
from flask_restful import Api

from app.routes.correlation_with_eng_level import \
    CorrelationWithEngagementLevel
from app.routes.correlation_with_score import CorrelationWithScore
from app.routes.correlation_without_score import CorrelationWithoutScore
from app.routes.distribution_graph import DistributionGraph
from app.routes.engagement_form_data import EngagementPredictionForm
from app.routes.engagement_prediction import EngagementPrediction
from app.routes.engagement_training import EngagementTraining
from app.routes.home import Home
from app.routes.with_score_prediction import WithScorePrediction
from app.routes.with_score_training import WithScoreTraining
from app.routes.without_score_prediction import WithoutScorePrediction
from app.routes.without_score_training import WithoutScoreTraining

app = Flask(__name__, template_folder='templates')
api = Api(app)

Home.register(app, route_base='/')

DistributionGraph.register(app, route_base='/distribution-graph')

CorrelationWithScore.register(app, route_base='/correlation-graph/with-score')

CorrelationWithoutScore.register(
    app, route_base='/correlation-graph/without-score')

CorrelationWithEngagementLevel.register(
    app, route_base='/correlation-graph/for-engagement-level')

EngagementTraining.register(app,  route_base='/training/for-engagement-level')

WithScoreTraining.register(app, route_base='/training/with-score/')

WithoutScoreTraining.register(app, route_base='/training/without-score/')

EngagementPredictionForm.register(app, route_base='/eng-form')

EngagementPrediction.register(app, route_base='/score-pred')

WithScorePrediction.register(app, route_base='/prediction-withscore')

WithoutScorePrediction.register(app, route_base='/prediction-withoutscore')


if __name__ == '__main__':
    app.run(debug=True, port=5005)
