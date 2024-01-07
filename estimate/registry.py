from estimate.models import EndPoint, MagModel, MagStatus


class MagRegistry:
    def __init__(self):
        self.endpoints = {}

    def add_model(self, endpoint_name, model_object, model_name, model_status,
                  model_version, owner, model_description, model_code):
        # get endpoint
        endpoint, _ = EndPoint.objects.get_or_create(name=endpoint_name, owner=owner)

        if not MagModel.objects.filter(name=model_name).exists():
            # print("不存在")
            database_object, model_created = MagModel.objects.get_or_create(
                name=model_name,
                description=model_description,
                code=model_code,
                version=model_version,
                owner=owner,
                parent_endpoint=endpoint)
            if model_created:
                status = MagStatus(status=model_status,
                                   created_by=owner,
                                   parent_model=database_object,
                                   active=True)
                status.save()
                database_object.save()

        else:
            # print("存在")
            database_object, _ = MagModel.objects.get_or_create(name=model_name)
            # print("database_object = {}".format(database_object))

        # add to registry
        self.endpoints[database_object.id] = model_object
        return None
