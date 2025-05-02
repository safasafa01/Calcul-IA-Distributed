from django.urls import path, include

urlpatterns = [
    path('', include('app.urls')),  # ← inclut toutes les routes définies dans app/urls.py
]