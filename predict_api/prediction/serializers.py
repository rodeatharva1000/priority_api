from rest_framework import serializers


class EquipmentInputSerializer(serializers.Serializer):
    equipment_category = serializers.IntegerField()
    equipment_farm_type = serializers.IntegerField()
    equipment_quality = serializers.IntegerField()
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()


class PredictionRequestSerializer(serializers.Serializer):
    user_category = serializers.IntegerField()
    user_farm_type = serializers.IntegerField()
    user_latitude = serializers.FloatField()
    user_longitude = serializers.FloatField()
    equipments = EquipmentInputSerializer(many=True)
    request_id = serializers.CharField(required=False)
