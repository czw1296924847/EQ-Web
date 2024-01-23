from django.urls import re_path
from estimate.consumers import *

websocket_urlpatterns = [
    re_path(r'ws/job-status/$', ChatConsumer.as_asgi()),
]

