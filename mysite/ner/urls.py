from django.urls import path

from . import views

urlpatterns = [
    path('', views.ner, name='ner'),
]