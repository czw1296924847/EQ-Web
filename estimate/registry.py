from .models import *


class MagRegistry:
    """
    Initialize Magnitude Estimation Model, when starting service
    """
    def __init__(self):
        self.models = {}
        self.users = {}

    def add_model(self, model_object, model_name, model_status,
                  model_version, model_owner, model_situation, model_description, model_code):

        if not MagModel.objects.filter(name=model_name).exists():
            database_object, model_created = MagModel.objects.get_or_create(
                name=model_name,
                description=model_description,
                code=model_code,
                version=model_version,
                owner=model_owner,
                situation=model_situation,
            )
            if model_created:
                status = MagStatus(status=model_status,
                                   created_by=model_owner,
                                   parent_model=database_object,
                                   active=True)
                status.save()
                database_object.save()

        else:
            database_object, _ = MagModel.objects.get_or_create(name=model_name)

        self.models[database_object.id] = model_object
        print("Successfully Load Model: {}".format(model_name))
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
        if not FeatureModel.objects.filter(param=param, description=description).exists():
            FeatureModel.objects.create(param=param, description=description)
        return None

    def init_info(self):
        """
        Initialize some information
        :return:
        """
        MagModel.objects.all().update(situation="Free")
        return None
