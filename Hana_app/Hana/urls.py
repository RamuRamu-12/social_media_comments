from django.urls import path
from .views import *

urlpatterns = [
    path('txt_response', gen_txt_response, name='processing_files'),
    path('graph response', gen_graph_response, name='query system'),
    path('txt_graph_response', gen_response, name='query system'),
]