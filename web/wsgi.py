"""
WSGI config for web project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import inspect
import estimate.network as network
import func.net as net
from estimate.registry import MagRegistry

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')

application = get_wsgi_application()

"""
registry user and model, for operation estimation
"""
registry = MagRegistry()

registry.add_model(model_object=network.MagInfoNet(),
                   model_name="MagInfoNet",
                   model_status="Production",
                   model_version="0.0.1",
                   model_owner="Chen Ziwei",
                   model_situation="Free",
                   model_description="Proposed model",
                   model_code=inspect.getsource(net.MagInfoNet))

registry.add_model(model_object=network.EQGraphNet(),
                   model_name="EQGraphNet",
                   model_status="Production",
                   model_version="0.0.1",
                   model_owner="Chen Ziwei",
                   model_situation="Free",
                   model_description="Proposed model",
                   model_code=inspect.getsource(net.EQGraphNet))

registry.add_model(model_object=network.MagNet(),
                   model_name="MagNet",
                   model_status="Production",
                   model_version="0.0.1",
                   model_owner="Mousavi",
                   model_situation="Free",
                   model_description="From doi.org/10.1029/2019GL085976",
                   model_code=inspect.getsource(net.MagNet))

registry.add_model(model_object=network.CREIME(),
                   model_name="CREIME",
                   model_status="Production",
                   model_version="0.0.1",
                   model_owner="Chakraborty",
                   model_situation="Free",
                   model_description="From doi.org/10.1029/2022JB024595",
                   model_code=inspect.getsource(net.CREIME))

registry.add_model(model_object=network.ConvNetQuakeINGV(),
                   model_name="ConvNetQuakeINGV",
                   model_status="Production",
                   model_version="0.0.1",
                   model_owner="Lomax",
                   model_situation="Free",
                   model_description="From article-abstract/90/2A/517/568771",
                   model_code=inspect.getsource(net.ConvNetQuakeINGV))


registry.add_user(username="czw",
                  password="fff")


registry.add_feature(param="source_magnitude",
                     description="Released seismic energy of earthquake")

registry.add_feature(param="source_depth_km",
                     description="Distance from ground to earthquake source")

registry.add_feature(param="source_distance_km",
                     description="Distance from observation to earthquake source")

registry.add_feature(param="snr_db",
                     description="Signal-to-noise ratio")

registry.add_feature(param="p_arrival_sample",
                     description="P wave arrival time")

registry.add_feature(param="s_arrival_sample",
                     description="S wave arrival time")

registry.init_info()

print()
