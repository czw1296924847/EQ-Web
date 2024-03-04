from .models import *


class DlRegistry:
    """
    Initialize Magnitude Estimation Model, when starting service
    """

    def __init__(self):
        self.models = {}
        self.users = {}

    def add_model(self, model_object, name, description, owner, path_data, library,
                  code_data, code_model, code_train, code_test, code_run):

        if not DlModel.objects.filter(name=name).exists():
            database_object, created = DlModel.objects.get_or_create(
                name=name,
                description=description,
                owner=owner,
                path_data=path_data,
                library=library,
                code_data=code_data,
                code_model=code_model,
                code_train=code_train,
                code_test=code_test,
                code_run=code_run,
            )
            if created:
                status = DlModelStatus(name=name, process="")
                status.save()
                database_object.save()

        else:
            database_object, _ = DlModel.objects.get_or_create(name=name)

        self.models[database_object.id] = model_object
        print("Successfully Load Model: {}".format(name))
        return None

    def add_user(self, username, password):
        """
        Initialize user, when starting service
        """
        if not User.objects.filter(username=username, password=password).exists():
            User.objects.create(username=username, password=password)
        self.users[username] = password
        print("Successfully Load User: {}".format(username))
        return None

    def add_feature(self, param, description):
        """
        add seismic feature in chunk.csv
        """
        if not Feature.objects.filter(param=param, description=description).exists():
            Feature.objects.create(param=param, description=description)
        return None

    def init_info(self):
        """
        Initialize model information
        """
        DlModel.objects.all().update(situation="Free")
        # ModelStatus.objects.all().update(process="")
        return None
