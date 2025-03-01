from django.urls import path
from .views import *

urlpatterns = [
    path('txt_response', gen_txt_response, name='processing_files'),
    path('sla_data_process', process_query, name='processing_files'),
    path('sla_query_making', sla_query, name="sla_query"),
]
