from django.urls import path

#from .views import SignUpView, run_detection
from . import views

urlpatterns = [
    path("signup/", views.SignUpView.as_view(), name="signup"),
    path("home/", views.DeteccionSomnolenciaView.as_view(), name="somnolencia"),
]

