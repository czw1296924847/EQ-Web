from rest_framework import serializers

from .models import *


class EndPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = EndPoint
        read_only_fields = ("id", "name", "owner", "created_at")
        fields = read_only_fields


class MagModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MagModel
        fields = ("pk", "name", "description", "code",
                  "version", "owner", "created_at", "situation")
        extra_kwargs = {
            'code': {'required': False},
            'version': {'required': False},
            'created_at': {'required': False},
            'situation': {'required': False},
            'process': {'required': False},
        }


class MagStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = MagStatus
        read_only_fields = ("id", "active")
        fields = ("id", "active", "status", "created_by", "created_at", "parent_model")


class MagRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MagRequest
        read_only_fields = ("id", "input_data", "full_response", "response", "created_at", "parent_model",)
        fields = ("id", "input_data", "full_response", "response", "feedback", "created_at", "parent_model",)


class FeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feature
        fields = ("pk", "param", "description")


class PointSerializer(serializers.Serializer):
    x = serializers.FloatField()
    y = serializers.FloatField()


class ResultSerializer(serializers.Serializer):
    points = PointSerializer(many=True)
    r2 = serializers.FloatField()
    rmse = serializers.FloatField()
    e_mean = serializers.FloatField()
    e_std = serializers.FloatField()
