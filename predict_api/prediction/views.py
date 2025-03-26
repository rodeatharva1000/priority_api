from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionRequestSerializer
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import math
import uuid

# Load the model and scaler when the server starts
# Update with your actual path
model = load_model('C:\\Users\\rodea\\Downloads\\predict_api\\model_requirements\\model.h5')
with open('C:\\Users\\rodea\\Downloads\\predict_api\\model_requirements\\scaler.pkl', 'rb') as f:  # Update with your actual path
    scaler = pickle.load(f)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


class EquipmentEvaluationView(APIView):
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        request_id = data.get('request_id', str(uuid.uuid4()))

        user_data = {
            'category': data['user_category'],
            'farm_type': data['user_farm_type'],
            'latitude': data['user_latitude'],
            'longitude': data['user_longitude']
        }

        results = []

        for idx, equipment in enumerate(data['equipments']):
            # Prepare input for model prediction
            input_data = np.array([[
                user_data['category'],
                user_data['farm_type'],
                equipment['equipment_category'],
                equipment['equipment_farm_type'],
                equipment['equipment_quality']
            ]])

            # Normalize the input data
            input_data_scaled = scaler.transform(input_data)

            # Predict the base points
            predicted_points = model.predict(input_data_scaled)[0][0]

            # Calculate distance
            distance = haversine(
                user_data['latitude'],
                user_data['longitude'],
                equipment['latitude'],
                equipment['longitude']
            )

            # Calculate additional points based on rules
            if user_data['category'] == equipment['equipment_category']:
                predicted_points += 200

            predicted_points += 150 * 1/(max(1, distance//10))

            results.append({
                'equipment_id': idx,
                'points': predicted_points,
                'distance_km': distance,
                'equipment_category': equipment['equipment_category'],
                'equipment_farm_type': equipment['equipment_farm_type'],
                'equipment_quality': equipment['equipment_quality'],
                'equipment_latitude': equipment['latitude'],
                'equipment_longitude': equipment['longitude']
            })

        # Sort results by points in descending order
        sorted_results = sorted(
            results, key=lambda x: x['points'], reverse=True)

        return Response({
            'request_id': request_id,
            'user_category': user_data['category'],
            'user_farm_type': user_data['farm_type'],
            'user_latitude': user_data['latitude'],
            'user_longitude': user_data['longitude'],
            'results': sorted_results,
            'top_recommendation': sorted_results[0] if sorted_results else None
        }, status=status.HTTP_200_OK)
