import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib.auth.models import User
from io import BytesIO
from django.http import JsonResponse
from .forms import RegisterForm, CSVUploadForm
from .models import CSVFile
import base64
from django.core.files.storage import FileSystemStorage

# Vue pour le login
def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')  # Rediriger les utilisateurs déjà connectés

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect")
            return redirect('login')
    
    return render(request, 'app/login.html')

# Vue pour le register
def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')

        if password != password_confirm:
            messages.error(request, "Les mots de passe ne correspondent pas.")
            return render(request, 'app/register.html')

        try:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            messages.success(request, "Inscription réussie ! Vous pouvez maintenant vous connecter.")
            return redirect('login')
        except Exception as e:
            messages.error(request, f"Erreur lors de l'inscription : {e}")
            return render(request, 'app/register.html')

    return render(request, 'app/register.html')

# Vue pour la déconnexion
def logout_view(request):
    logout(request)
    return redirect('login')

# Vue d'accueil protégée (seulement pour les utilisateurs connectés)
@login_required
def home(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['dataset']
        model_type = request.POST['model_type']

        fs = FileSystemStorage()
        file_name = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_name)

        # Ici tu peux déclencher le pré-traitement + entraînement
        return redirect('resultats_entrainement')  # On créera cette page après

    return render(request, 'app/home.html')




@login_required
def resultats_entrainement(request):
    return HttpResponse("Résultats de l'entraînement à afficher ici.")




