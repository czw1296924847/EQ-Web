from django.urls import include, re_path, path
from rest_framework.routers import DefaultRouter

from .views import *

app_name = "weather"

router = DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),

    re_path(r'^history24hour', History24HourView.as_view()),
    re_path(r'^future7day', Future7DayView.as_view()),
]
