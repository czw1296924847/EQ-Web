from rest_framework import serializers

from estimate.models import EndPoint, MagModel, MagStatus, MagRequest


class EndPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = EndPoint
        read_only_fields = ("id", "name", "owner", "created_at")
        fields = read_only_fields


class MagModelSerializer(serializers.ModelSerializer):
    current_status = serializers.SerializerMethodField(read_only=True)

    def get_current_status(self, model):
        return MagStatus.objects.filter(parent_model=model).latest('created_at').status

    class Meta:
        model = MagModel
        read_only_fields = ("id", "name", "description", "code",
                            "version", "owner", "created_at",
                            "parent_endpoint", "current_status")
        fields = read_only_fields


class MagStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = MagStatus
        read_only_fields = ("id", "active")
        fields = ("id", "active", "status", "created_by", "created_at", "parent_model")


class MagRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MagRequest
        read_only_fields = (
            "id",
            "input_data",
            "full_response",
            "response",
            "created_at",
            "parent_model",
        )
        fields = (
            "id",
            "input_data",
            "full_response",
            "response",
            "feedback",
            "created_at",
            "parent_model",
        )
