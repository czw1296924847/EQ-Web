from django.urls import include, re_path, path
from rest_framework.routers import DefaultRouter

from .views import *

app_name = "estimate"

router = DefaultRouter()
router.register(r"dl_model", DlModelViewSet, basename="dl_model")
router.register(r"dl_model_status", DlModelStatusViewSet, basename="dl_model_status")
router.register(r"feature", FeatureViewSet, basename="feature")

urlpatterns = [
    path('', include(router.urls)),

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
]
