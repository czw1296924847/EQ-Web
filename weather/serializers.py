from rest_framework.serializers import *


class CitySerializer(Serializer):
    city = CharField()
    date = ListField(child=CharField())
    temp = ListField(child=FloatField(), allow_null=True)
    wea = ListField(child=CharField(), allow_null=True)
    win = ListField(child=CharField(), allow_null=True)
    win_s = ListField(child=FloatField(), allow_null=True)
    ppt = ListField(child=FloatField(), allow_null=True)
    humid = ListField(child=FloatField(), allow_null=True)
