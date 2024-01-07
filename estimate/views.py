from rest_framework.exceptions import APIException
from rest_framework import viewsets
from rest_framework import mixins
from django.db import transaction
import json
import pandas as pd
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response

from estimate.serializers import EndPointSerializer, MagModelSerializer, MagStatusSerializer, MagRequestSerializer
from estimate.models import EndPoint, MagModel, MagStatus, MagRequest
from estimate.registry import MagRegistry


def get_post_data(request, endpoint_name, model_status, model_version):
    from web.wsgi import registry
    models = MagModel.objects.filter(parent_endpoint__name=endpoint_name, status__status=model_status,
                                     status__active=True)
    if model_version is not None:
        models = models.filter(version=model_version)
    # if len(models) != 5:
    #     return Response({"status": "Error", "message": "MagModels have errors. Please remove db and generate it."},
    #                     status=status.HTTP_400_BAD_REQUEST, )
    network_name = pd.DataFrame(request.data, index=[0])['network'].values[0]
    model = models.filter(name=network_name)[0]
    if not model:
        return Response({"status": "Error", "message": "MagModel is not available"},
                        status=status.HTTP_400_BAD_REQUEST, )
    print("model id = {}".format(model.id))
    model_object = registry.endpoints[model.id]
    print("model_object = {}".format(model_object))
    return model, model_object


class EndPointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = EndPointSerializer
    queryset = EndPoint.objects.all()


class ModelViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    pass


class StatusViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.CreateModelMixin):
    pass


class RequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.UpdateModelMixin):
    pass


class MagModelViewSet(ModelViewSet):
    serializer_class = MagModelSerializer
    queryset = MagModel.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MagStatus.objects.filter(parent_model=instance.parent_model,
                                            created_at__lt=instance.created_at, active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MagStatus.objects.bulk_update(old_statuses, ["active"])


class MagStatusViewSet(StatusViewSet):
    serializer_class = MagStatusSerializer
    queryset = MagStatus.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                deactivate_other_statuses(instance)
        except Exception as e:
            raise APIException(str(e))


class MagRequestViewSet(RequestViewSet):
    serializer_class = MagRequestSerializer
    queryset = MagRequest.objects.all()


# model train
class ModelTrainView(views.APIView):
    def post(self, request, endpoint_name):
        model_status = self.request.query_params.get("status", "production")
        model_version = self.request.query_params.get("version")

        model, model_object = get_post_data(request, endpoint_name, model_status, model_version)
        result_train = model_object.training(request.data)

        mag_request = MagRequest(
            input_data=json.dumps(request.data),
            full_response=result_train,
            response="",
            feedback="",
            parent_model=model,
        )
        mag_request.save()
        return Response(result_train)


# model test
class ModelTestView(views.APIView):
    def post(self, request, endpoint_name):
        model_status = self.request.query_params.get("status", "production")
        model_version = self.request.query_params.get("version")

        model, model_object = get_post_data(request, endpoint_name, model_status, model_version)
        result_test = model_object.testing(request.data)

        mag_request = MagRequest(
            input_data=json.dumps(request.data),
            full_response=result_test,
            response="",
            feedback="",
            parent_model=model,
        )
        mag_request.save()
        return Response(result_test)

