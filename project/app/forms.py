from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from .models import CSVFile

class RegisterForm(forms.ModelForm):
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirmer le mot de passe")

    class Meta:
        model = User
        fields = ['username', 'password']

    def clean_password2(self):
        password1 = self.cleaned_data.get("password")
        password2 = self.cleaned_data.get("password2")

        if password1 and password2 and password1 != password2:
            raise ValidationError("Les mots de passe ne correspondent pas")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)  # Crée l'objet User, sans l'enregistrer immédiatement
        user.set_password(self.cleaned_data['password'])  # Définit le mot de passe crypté
        if commit:
            user.save()  # Sauvegarde l'utilisateur dans la base de données
        return user


class CSVUploadForm(forms.Form):
    fichier_csv = forms.FileField(label='Fichier CSV')