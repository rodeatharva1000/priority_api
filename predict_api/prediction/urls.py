from django.urls import path
from .views import EquipmentEvaluationView

urlpatterns = [
    path('evaluate-equipments/', EquipmentEvaluationView.as_view(),
         name='evaluate-equipments'),
]
