from django.urls import path
from .main_app import eapl_flow, eapl_flow_get, eapl_flow_post

urlpatterns = [
    path('eapl-nlp/jobstatus/<str:job_id>', eapl_flow_get),
    path('eapl-nlp/', eapl_flow),
    path('eapl-nlp/<str:api_name>', eapl_flow_post),
    path('eapl-nlp/<str:api_name>/<str:version>', eapl_flow_post),
]
