from django.urls import include, re_path, path

from .views import *

app_name = "estimate"

urlpatterns = [
    re_path(r'^run$', RunView.as_view()),
    re_path(r'^models$', ModelListView.as_view()),
    re_path(r'^features$', FeatureListView.as_view()),
    re_path(r'^features/dist$', FeatureDistView.as_view()),
    re_path(r'^features/locate$', FeatureLocateView.as_view()),
    re_path(r'models/([0-9]*)$', ModelOptView.as_view()),
    re_path(r'^(?P<model_name>.+)/train$', ModelTrainView.as_view()),
    re_path(r'^(?P<model_name>.+)/test$', ModelTestView.as_view()),
    re_path(r'^(?P<model_name>.+)/detail$', ModelDetailView.as_view()),
    re_path(r'^(?P<model_name>.+)/process', ModelProcessView.as_view()),
    re_path(r'^(?P<model_name>.+)/(?P<opt>.+)/true_pred$', CompTruePredView.as_view()),
    re_path(r'^(?P<model_name>.+)/(?P<opt>.+)/loss$', LossCurveView.as_view()),
    re_path(r'^(?P<model_name>.+)/(?P<opt>.+)/record$', ModelRecordView.as_view()),
    re_path(r'^login$', LoginView.as_view()),

    path('', index, name='index'),
    path('<str:room_name>/', room, name='room'),
]
