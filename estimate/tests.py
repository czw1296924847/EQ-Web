import torch
import numpy as np
import os
import inspect
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
django.setup()
from django.test import TestCase
from estimate.network import MagNet
from estimate.registry import MagRegistry


class DlTests(TestCase):
    def test_magnet(self):
        device = "cuda:1"
        x = torch.rand(1, 3, 6000).float().to(device)
        Mag = MagNet()
        Mag.model.to(device)
        y = Mag.model(x)
        self.assertEquals(torch.is_tensor(y), True)

    def test_registry(self):
        registry = MagRegistry()
        self.assertEquals(len(registry.endpoints), 0)
        endpoint_name = "magnitude_estimator"
        model_object = MagNet()
        model_name = "MagNet"
        model_status = "production"
        model_version = "0.0.1"
        model_owner = "Chen Ziwei"
        model_description = "From doi.org/10.1029/2019GL085976"
        model_code = inspect.getsource(MagNet)
        registry.add_model(endpoint_name, model_object, model_name, model_status,
                           model_version, model_owner, model_description, model_code)
        self.assertEquals(len(registry.endpoints), 1)
