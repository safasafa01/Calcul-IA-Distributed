from django.db import models
from django.contrib.auth.models import User

class CSVFile(models.Model):
    utilisateur = models.ForeignKey(User, on_delete=models.CASCADE)
    fichier = models.FileField(upload_to='uploads/')
    date_televersement = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.fichier.name} ({self.utilisateur.username})"