import torch
import numpy as np
import os
import inspect
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
django.setup()
from django.test import TestCase
from estimate.network import MagNet
from estimate.registry import DlRegistry
from func.process import get_lib_by_files, duplicate_lib, get_source
from func.net import MagInfoNet


class DlTests(TestCase):
    def test_magnet(self):
        device = "cuda:1"
        x = torch.rand(1, 3, 6000).float().to(device)
        Mag = MagNet()
        Mag.model.to(device)
        y = Mag.model(x)
        self.assertEquals(torch.is_tensor(y), True)

    def test_registry(self):
        registry = DlRegistry()
        model_object = MagNet()
        name = "MagNet"
        owner = "Mou"
        path_data = ""
        library = ""
        code_data = ""
        description = "From doi.org/10.1029/2019GL085976"
        code_model = ""
        code_train = ""
        code_test = ""
        registry.add_model(model_object, name, description, owner, path_data, library,
                           code_data, code_model, code_train, code_test)
        self.assertEquals(len(registry.models), 1)

    def test_get_library_files(self):
        files = ['network.py']

        library = get_lib_by_files(files, True)
        self.assertIn("sys.path.append(\"..\")", library.split('\n'))

        library = get_lib_by_files(files, False)
        self.assertIn("class Net:", library.split('\n'))

    def test_duplicate_lib(self):
        string = "import numpy as np\nimport numpy as np"
        library = duplicate_lib(string)
        self.assertEquals(library, "import numpy as np")

    def test_get_source(self):

        def code_data():
            def data():
                training_data = datasets.FashionMNIST(
                    root="data",
                    train=True,
                    download=True,
                    transform=ToTensor(),
                )

        string_true = """def data():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )"""
        string_out = get_source(code_data, True)
        self.assertEquals(string_true, string_out)

        def code_train():
            training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor(),
            )

        string_true = """training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)"""
        string_out = get_source(code_train, True)
        self.assertEquals(string_true, string_out)
