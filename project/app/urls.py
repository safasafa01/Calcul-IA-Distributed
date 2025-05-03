from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name='home'), 
    path('resultats/', views.success, name='resultats'),
    path('accueil/', views.accueil, name='accueil'),
    path('parametres/', views.parametres, name='parametres'),
    path('lancement/', views.lancement, name='lancement'),
    path('statistiques/', views.statistiques, name='statistiques'),
   
]