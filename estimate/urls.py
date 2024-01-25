from django.urls import include, re_path, path
from rest_framework.routers import DefaultRouter

from .views import *

app_name = "estimate"

urlpatterns = [
    # re_path(r"^(?P<endpoint_name>.+)/test$", ModelTestView.as_view(), name="test"),
    # re_path(r"^(?P<endpoint_name>.+)/train$", ModelTrainView.as_view(), name="train"),
    re_path(r'^magnitude/models$', ModelListView.as_view()),
    re_path(r'^magnitude/features$', FeatureListView.as_view()),
    re_path(r'^magnitude/features/dist$', FeatureDistView.as_view()),
    re_path(r'magnitude/models/([0-9])$', ModelOptView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/train$', ModelTrainOneView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/test$', ModelTestOneView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/detail$', ModelDetailView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/process', ModelProcessView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/(?P<opt>.+)/true_pred$', CompTruePredView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/(?P<opt>.+)/loss$', LossCurveView.as_view()),
    re_path(r'^magnitude/(?P<model_name>.+)/(?P<opt>.+)/record$', ModelRecordView.as_view()),
    re_path(r'^magnitude/login$', LoginView.as_view()),

    path('', index, name='index'),
    path('<str:room_name>/', room, name='room'),
]

# router = DefaultRouter(trailing_slash=False)
# router.register(r"EndPoint", EndPointViewSet, basename="EndPoint")
# router.register(r"Model", MagModelViewSet, basename="Model")
# router.register(r"Status", MagStatusViewSet, basename="Status")
# router.register(r"Request", MagRequestViewSet, basename="Request")
#
# urlpatterns += router.urls
