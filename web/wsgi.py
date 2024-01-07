"""
WSGI config for web project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import inspect
from estimate.network import MagInfoNet, EQGraphNet, MagNet, CREIME, ConvNetQuakeINGV
from estimate.registry import MagRegistry

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')

application = get_wsgi_application()


"""
registry model
"""
registry = MagRegistry()

MaI = MagInfoNet()
registry.add_model(endpoint_name="magnitude_estimator",
                   model_object=MaI,
                   model_name="MagInfoNet",
                   model_status="production",
                   model_version="0.0.1",
                   owner="Chen Ziwei",
                   model_description="Proposed model",
                   model_code=inspect.getsource(MagInfoNet))
print("registry.endpoints = {}".format(registry.endpoints))

EQG = EQGraphNet()
registry.add_model(endpoint_name="magnitude_estimator",
                   model_object=EQG,
                   model_name="EQGraphNet",
                   model_status="production",
                   model_version="0.0.1",
                   owner="Chen Ziwei",
                   model_description="Proposed model",
                   model_code=inspect.getsource(EQGraphNet))
print("registry.endpoints = {}".format(registry.endpoints))

Mag = MagNet()
registry.add_model(endpoint_name="magnitude_estimator",
                   model_object=Mag,
                   model_name="MagNet",
                   model_status="production",
                   model_version="0.0.1",
                   owner="Mousavi",
                   model_description="From doi.org/10.1029/2019GL085976",
                   model_code=inspect.getsource(MagNet))
print("registry.endpoints = {}".format(registry.endpoints))

CRE = CREIME()
registry.add_model(endpoint_name="magnitude_estimator",
                   model_object=CRE,
                   model_name="CREIME",
                   model_status="production",
                   model_version="0.0.1",
                   owner="Chakraborty",
                   model_description="From doi.org/10.1029/2022JB024595",
                   model_code=inspect.getsource(CREIME))
print("registry.endpoints = {}".format(registry.endpoints))

COI = ConvNetQuakeINGV()
registry.add_model(endpoint_name="magnitude_estimator",
                   model_object=COI,
                   model_name="ConvNetQuakeINGV",
                   model_status="production",
                   model_version="0.0.1",
                   owner="Lomax",
                   model_description="From pubs.geoscienceworld.org/ssa/srl/article-abstract/90/2A/517/568771",
                   model_code=inspect.getsource(ConvNetQuakeINGV))
print("registry.endpoints = {}".format(registry.endpoints))

