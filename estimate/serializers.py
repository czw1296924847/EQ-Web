from rest_framework import serializers

from .models import *


class DlModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = DlModel
        fields = ('pk', 'name', 'description', 'version', 'owner', 'created_at', 'situation', 'path_data',
                  'library', 'code_data', 'code_model', 'code_train', 'code_test', 'code_run')
        extra_kwargs = {
            'version': {'required': False},
            'created_at': {'required': False},
            'situation': {'required': False},
            'path_data': {'required': False},
            'code_data': {'required': False},
            'library': {'required': False},
            'code_model': {'required': False},
            'code_train': {'required': False},
            'code_test': {'required': False},
            'code_run': {'required': False},
        }


class DlModelStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = DlModelStatus
        fields = ('pk', 'name', 'process')


class FeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feature
        fields = ('pk', 'param', 'description')


class PointSerializer(serializers.Serializer):
    x = serializers.FloatField()
    y = serializers.FloatField()


class SourceSerializer(serializers.Serializer):
    Longitude = serializers.FloatField()
    Latitude = serializers.FloatField()
    Magnitude = serializers.FloatField()


class ResultSerializer(serializers.Serializer):
    points = PointSerializer(many=True)
    r2 = serializers.FloatField()
    rmse = serializers.FloatField()
    e_mean = serializers.FloatField()
    e_std = serializers.FloatField()


class LibSerializer(serializers.Serializer):
    name = serializers.CharField(allow_blank=True)
    version = serializers.CharField(allow_blank=True)


class CondaSerializer(serializers.Serializer):
    env = serializers.CharField()
    lib = LibSerializer(many=True)
