"""
WSGI config for web project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

from inspect import getsource
import estimate.network as network
import estimate.views as views
import func.net as net
from func.process import ROOT, DATA_AD
from func.process import get_lib_by_files, get_source
from estimate.registry import DlRegistry
from estimate.static.detail import MagInfoNet, EQGraphNet, MagNet, CREIME, ConvNetQuakeINGV, TestNet

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')

application = get_wsgi_application()

"""
registry user and model, for operation estimation
"""
registry = DlRegistry()

registry.add_model(model_object=network.MagInfoNet(),
                   name="MagInfoNet",
                   description="Proposed model",
                   owner="Chen Ziwei",
                   path_data=ROOT,
                   library=get_source(MagInfoNet.code_lib),
                   code_data=get_source(MagInfoNet.code_data),
                   code_model=get_source(MagInfoNet.MagInfoNet, False),
                   code_train=get_source(MagInfoNet.code_train),
                   code_test=get_source(MagInfoNet.code_test),
                   code_run=get_source(MagInfoNet.code_run),)

registry.add_model(model_object=network.EQGraphNet(),
                   name="EQGraphNet",
                   description="Proposed model",
                   owner="Chen Ziwei",
                   path_data=ROOT,
                   library=get_source(EQGraphNet.code_lib),
                   code_data=get_source(EQGraphNet.code_data),
                   code_model=get_source(EQGraphNet.EQGraphNet, False),
                   code_train=get_source(EQGraphNet.code_train),
                   code_test=get_source(EQGraphNet.code_test),
                   code_run=get_source(EQGraphNet.code_run),)

registry.add_model(model_object=network.MagNet(),
                   name="MagNet",
                   description="From 10.1029/2019GL085976",
                   owner="Mousavi",
                   path_data=ROOT,
                   library=get_source(MagNet.code_lib),
                   code_data=get_source(MagNet.code_data),
                   code_model=get_source(MagNet.MagNet, False),
                   code_train=get_source(MagNet.code_train),
                   code_test=get_source(MagNet.code_test),
                   code_run=get_source(MagNet.code_run),)

registry.add_model(model_object=network.CREIME(),
                   name="CREIME",
                   description="From 10.1029/2022JB024595",
                   owner="Chakraborty",
                   path_data=ROOT,
                   library=get_source(CREIME.code_lib),
                   code_data=get_source(CREIME.code_data),
                   code_model=get_source(CREIME.CREIME, False),
                   code_train=get_source(CREIME.code_train),
                   code_test=get_source(CREIME.code_test),
                   code_run=get_source(CREIME.code_run),)

registry.add_model(model_object=network.ConvNetQuakeINGV(),
                   name="ConvNetQuakeINGV",
                   description="From 90/2A/517/568771",
                   owner="Lomax",
                   path_data=ROOT,
                   library=get_source(ConvNetQuakeINGV.code_lib),
                   code_data=get_source(ConvNetQuakeINGV.code_data),
                   code_model=get_source(ConvNetQuakeINGV.ConvNetQuakeINGV, False),
                   code_train=get_source(ConvNetQuakeINGV.code_train),
                   code_test=get_source(ConvNetQuakeINGV.code_test),
                   code_run=get_source(ConvNetQuakeINGV.code_run),)

registry.add_model(model_object=None,
                   name="TestNet",
                   description="自定义模型示例，上传代码、运行",
                   owner="Chen Ziwei",
                   path_data=DATA_AD,
                   library=get_source(TestNet.code_lib),
                   code_data=get_source(TestNet.code_data),
                   code_model=get_source(TestNet.TestNet, False),
                   code_train=get_source(TestNet.code_train),
                   code_test=get_source(TestNet.code_test),
                   code_run=get_source(TestNet.code_run))

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

registry.add_feature(param="source_latitude",
                     description="Latitude of earthquake source")

registry.add_feature(param="source_longitude",
                     description="Longitude of earthquake source")

registry.init_info()

print()
