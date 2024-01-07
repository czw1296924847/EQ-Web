from django.urls import include, re_path
from rest_framework.routers import DefaultRouter

from estimate.views import EndPointViewSet, MagModelViewSet, MagStatusViewSet, MagRequestViewSet, \
    ModelTestView, ModelTrainView


router = DefaultRouter(trailing_slash=False)
router.register(r"EndPoint", EndPointViewSet, basename="EndPoint")
router.register(r"Model", MagModelViewSet, basename="Model")
router.register(r"Status", MagStatusViewSet, basename="Status")
router.register(r"Request", MagRequestViewSet, basename="Request")

urlpatterns = [
    re_path(r"^", include(router.urls)),
    re_path(r"^(?P<endpoint_name>.+)/test$", ModelTestView.as_view(), name="test"),
    re_path(r"^(?P<endpoint_name>.+)/train$", ModelTrainView.as_view(), name="train"),
]
