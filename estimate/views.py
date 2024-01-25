from rest_framework.exceptions import APIException
from rest_framework.viewsets import GenericViewSet
from rest_framework.mixins import RetrieveModelMixin, ListModelMixin, CreateModelMixin, UpdateModelMixin
from rest_framework import views, status, generics
from rest_framework.response import Response
from django.db import transaction
from django.shortcuts import render
from celery import shared_task
import json
import pandas as pd
import numpy as np
import os
import os.path as osp
from .serializers import *
from .models import *
from func.process import ROOT, RE_AD, get_dist
from func.net import cal_metrics


def get_model(pk):
    try:
        model = MagModel.objects.get(pk=pk)
    except MagModel.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    return model


def get_record(opt):
    model_names = list(MagModel.objects.values_list('name', flat=True))  # Model Name
    record = []
    for model_name in model_names:
        path = osp.join(RE_AD, model_name)
        data_sizes = [f for f in os.listdir(path) if osp.isdir(osp.join(path, f))]  # Data Size
        record_one_model = []
        for data_size in data_sizes:
            files = os.listdir(osp.join(path, str(data_size)))  # File name
            for file in files:
                params = file.split('_')
                if (params[0] != opt) or (params[1] != 'true'):
                    continue
                sm_scale, chunk_name, data_size_train = params[2], params[3], int(params[4])
                train_ratio = np.round(data_size_train / int(data_size), 2)
                record_one_file = {
                    "train_ratio": str(train_ratio),
                    "data_size": str(data_size),
                    "sm_scale": sm_scale,
                    "chunk_name": chunk_name
                }
                record_one_model.append(record_one_file)
        record.append({'model_name': model_name, 'record': record_one_model})
    return record


class ModelTrainOneView(views.APIView):
    def get(self, request, model_name):
        """
        Get trained model information

        :param model_name:
        :param request:
        :return:
        """
        return Response(get_record("train"))

    def post(self, request, model_name):
        """
        Train model

        :param request:
        :param model_name: Model name, like: MagInfoNet, EQGraphNet, MagNet,
        :return: Train result, given by network.cal_metrics
        """
        from web.wsgi import registry
        model = MagModel.objects.filter(name=model_name)[0]

        if model.situation == "testing":
            return Response({"error": "Is testing"}, status=status.HTTP_409_CONFLICT)
        model.situation = "training"
        model.save()

        model_object = registry.models[model.id]
        result_train = model_object.training(request.data, model_name)
        model.situation = "Free"
        model.save()
        return Response(result_train)


class ModelTestOneView(views.APIView):
    def get(self, request, model_name):
        """
        Get trained model information

        :param model_name:
        :param request:
        :return:
        """
        return Response(get_record("test"))

    def post(self, request, model_name):
        """
        Test model

        :param request:
        :param model_name: Model name, like: MagInfoNet, EQGraphNet, MagNet,
        :return: Test result, given by network.cal_metrics
        """
        from web.wsgi import registry
        model = MagModel.objects.filter(name=model_name)[0]
        if model.situation == "training":
            return Response({"error": "Is training"}, status=status.HTTP_409_CONFLICT)
        model.situation = "testing"
        model.save()

        model_object = registry.models[model.id]
        try:
            result_test = model_object.testing(request.data, model_name)
            model.situation = "Free"
            model.save()
            return Response(result_test)
        except FileNotFoundError:
            model.situation = "Free"
            model.save()
            return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)


class ModelListView(views.APIView):
    def get(self, request):
        """
        show all models, from ModelList.js

        :param request:
        :return:
        """
        model_list = MagModel.objects.all()
        serializer = MagModelSerializer(model_list, many=True, context={'request': request})
        return Response(serializer.data)

    def post(self, request):
        """
        create new model, from ModelList.js

        :param request:
        :return:
        """
        serializer = MagModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FeatureListView(views.APIView):
    def get(self, request):
        """
        show all params from data set, from FeatureList.js

        :param request:
        :return:
        """
        features = Feature.objects.all()
        serializer = FeatureSerializer(features, many=True, context={'request': request})
        return Response(serializer.data)


class FeatureDistView(views.APIView):
    def get(self, request):
        """
        get value dist of feature

        :param request:
        :return:
        """
        v_min, v_max = None, None
        feature = request.GET.get('feature')
        bins = int(request.GET.get('bins'))
        chunk_name = request.GET.get('chunk_name')
        data_size = int(request.GET.get('data_size'))

        if feature == "source_depth_km":
            v_min, v_max = 0, 150
        elif feature == "source_magnitude":
            v_min = 0

        x, y = get_dist(feature, bins, chunk_name, data_size, v_min, v_max)

        if feature in ["source_distance_km", "source_depth_km", "snr_db",
                       "p_arrival_sample", "s_arrival_sample"]:
            x = np.round(x)
        elif feature == "source_magnitude":
            x = np.round(x, 2)

        points = [{"x": i, "y": j} for i, j in zip(x, y)]
        serializer = PointSerializer(points, many=True)
        return Response(serializer.data)


class ModelOptView(views.APIView):
    def put(self, request, pk):
        """
        Modify model information, from ModelList.js

        :param request:
        :param pk: Primary Key
        :return:
        """
        model = get_model(pk)
        print(model)
        serializer = MagModelSerializer(model, data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Delete model information, without used (danger)

        :param request:
        :param pk: Primary Key
        :return:
        """
        model = get_model(pk)
        model.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ModelDetailView(views.APIView):
    def get(self, request, model_name):
        """
        show detail information of a specific model, from ModelDetail.js

        :param request:
        :param model_name: Model name, like: MagInfoNet, EQGraphNet, MagNet,
        :return: Model information, given by models.MagModel
        """
        model = MagModel.objects.filter(name=model_name)
        serializer = MagModelSerializer(model, many=True, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ModelProcessView(views.APIView):
    def get(self, request, model_name):
        """
        get the situation during train/test process
        """
        process = ModelStatus.objects.values_list('process', flat=True).get(name=model_name)
        return Response(process, status=status.HTTP_200_OK)

    def put(self, request, model_name):
        """
        initialize the model process, to be ""
        """
        model = ModelStatus.objects.get(name=model_name)
        model.process = ""
        model.save()
        return Response(status=status.HTTP_200_OK)


def get_params(request):
    """
    Read params, from OptResult.js
    """
    train_ratio = float(request.GET.get('train_ratio'))
    data_size = int(request.GET.get('data_size'))
    sm_scale = request.GET.get('sm_scale')
    chunk_name = request.GET.get('chunk_name')
    data_size_train = int(data_size * train_ratio)
    data_size_test = data_size - data_size_train
    return sm_scale, chunk_name, data_size, data_size_train, data_size_test


def get_result_ad(re_ad, opt, sm_scale, chunk_name, data_size_train, data_size_test):
    """
    get result file address
    """
    true_ad = osp.join(re_ad, "{}_true_{}_{}_{}_{}.npy".
                       format(opt, sm_scale, chunk_name, str(data_size_train), str(data_size_test)))
    pred_ad = osp.join(re_ad, "{}_pred_{}_{}_{}_{}.npy".
                       format(opt, sm_scale, chunk_name, str(data_size_train), str(data_size_test)))
    loss_ad = osp.join(re_ad, "{}_loss_{}_{}_{}_{}.npy".
                       format(opt, sm_scale, chunk_name, str(data_size_train), str(data_size_test)))
    model_ad = osp.join(re_ad, "model_{}_{}_{}_{}.pkl".
                        format(sm_scale, chunk_name, str(data_size_train), str(data_size_test)))
    return true_ad, pred_ad, loss_ad, model_ad


class CompTruePredView(views.APIView):
    def get(self, request, model_name, opt, *args, **kwargs):
        """
        Compare the true and predicted magnitudes, from OptResult.js

        :param request:
        :param model_name: Model name
        :param opt: 'train' or 'test'
        :return: True magnitudes or Predicted operation
        """
        sm_scale, chunk_name, data_size, data_size_train, data_size_test = get_params(request)
        re_ad = osp.join(RE_AD, model_name, str(data_size))

        # set a smaller number for web show, to avoid web crash
        num_show = 500
        true_ad, pred_ad, _, _ = get_result_ad(re_ad, opt, sm_scale, chunk_name, data_size_train, data_size_test)
        true, pred = np.load(true_ad), np.load(pred_ad)
        points = [{"x": i, "y": j} for i, j in zip(true[:num_show], pred[:num_show])]
        r2, rmse, e_mean, e_std = cal_metrics(true, pred)
        data = {
            'points': points,
            'r2': str(np.round(float(r2), 4)),
            'rmse': str(np.round(float(rmse), 4)),
            'e_mean': str(np.round(float(e_mean), 4)),
            'e_std': str(np.round(float(e_std), 4)),
        }
        serializer = ResultSerializer(data)
        return Response(serializer.data)


class LossCurveView(views.APIView):
    def get(self, request, model_name, opt, *args, **kwargs):
        """
        Plot the loss curve during training, from OptResult.js

        :param request:
        :param model_name: Model name
        :param opt: 'train' or 'test'
        :return:
        """
        sm_scale, chunk_name, data_size, data_size_train, data_size_test = get_params(request)
        re_ad = osp.join(RE_AD, model_name, str(data_size))
        loss = np.load(osp.join(re_ad, "{}_loss_{}_{}_{}_{}.npy".
                                format(opt, sm_scale, chunk_name, str(data_size_train), str(data_size_test))))
        points = [{"x": i, "y": j} for i, j in zip(np.arange(loss.shape[0]), loss)]
        serializer = PointSerializer(points, many=True)
        return Response(serializer.data)


class ModelRecordView(views.APIView):
    def get(self, request, model_name, opt, *args, **kwargs):
        sm_scale, chunk_name, data_size, data_size_train, data_size_test = get_params(request)
        re_ad = osp.join(RE_AD, model_name, str(data_size))
        true_ad, pred_ad, loss_ad, model_ad = get_result_ad(re_ad, opt, sm_scale, chunk_name, data_size_train,
                                                            data_size_test)
        if opt == "train":
            return Response(
                osp.exists(true_ad) and osp.exists(pred_ad) and osp.exists(loss_ad) and osp.exists(model_ad),
                status=status.HTTP_200_OK)
        elif opt == "test":
            return Response(osp.exists(true_ad) and osp.exists(pred_ad) and osp.exists(loss_ad),
                            status=status.HTTP_200_OK)
        else:
            return Response(False, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, model_name, opt, *args, **kwargs):
        """
        Delete the train/test result (model.pkl, loss.npy, true.npy, pred.npy)

        :param request:
        :param model_name:
        :param opt:
        :return:
        """
        sm_scale, chunk_name, data_size, data_size_train, data_size_test = get_params(request)
        re_ad = osp.join(RE_AD, model_name, str(data_size))
        true_ad, pred_ad, loss_ad, model_ad = get_result_ad(re_ad, opt, sm_scale, chunk_name,
                                                            data_size_train, data_size_test)
        os.remove(true_ad), os.remove(pred_ad), os.remove(loss_ad)
        if opt == "train":
            os.remove(model_ad)
        return Response(status=status.HTTP_204_NO_CONTENT)


class LoginView(views.APIView):
    def get(self, request, *args, **kwargs):
        """
        check if use can log in
        """
        username = request.GET.get('username')
        password = request.GET.get('password')
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            print("username: {}, password: {}, 用户不存在".format(username, password))
            return Response({"msg": "user_not_exist"}, status=status.HTTP_404_NOT_FOUND)
        if user.password == password:
            print("username: {}, password: {}, 登录成功".format(username, password))
            return Response({"msg": "login_success"}, status=status.HTTP_200_OK)
        else:
            print("username: {}, password: {}, 密码错误".format(username, password))
            return Response({"msg": "password_error"}, status=status.HTTP_401_UNAUTHORIZED)


"""
Practice Code
"""
def get_post_data(request, endpoint_name, model_status, model_version):
    model_list = MagModel.objects.filter(parent_endpoint__name=endpoint_name, status__status=model_status,
                                         status__active=True)
    if model_version is not None:
        model_list = model_list.filter(version=model_version)
    # if len(model_list) != 5:
    #     return Response({"status": "Error", "message": "MagModels have errors. Please remove db and generate it."},
    #                     status=status.HTTP_400_BAD_REQUEST, )
    model_name = pd.DataFrame(request.data, index=[0])['model'].values[0]
    model = model_list.filter(model_name=model_name)[0]
    if not model:
        return Response({"status": "Error", "message": "MagModel is not available"},
                        status=status.HTTP_400_BAD_REQUEST, )
    model_object = registry.endpoints[model.id]
    return model, model_object


class StatusViewSet(RetrieveModelMixin, ListModelMixin, GenericViewSet, CreateModelMixin):
    pass


class RequestViewSet(RetrieveModelMixin, ListModelMixin, GenericViewSet, UpdateModelMixin):
    pass


class EndPointViewSet(RetrieveModelMixin, ListModelMixin, GenericViewSet):
    serializer_class = EndPointSerializer
    queryset = EndPoint.objects.all()


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


class ModelViewSet(RetrieveModelMixin, ListModelMixin, GenericViewSet):
    pass


class MagModelViewSet(ModelViewSet):
    serializer_class = MagModelSerializer
    queryset = MagModel.objects.all()


def index(request):
    return render(request, 'index.html', {})


def room(request, room_name):
    return render(request, 'room.html', {
        'room_name': room_name
    })
