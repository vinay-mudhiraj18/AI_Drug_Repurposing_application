from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    QUERY_TYPES = [
        ('drug', 'Drug'),
        ('protein', 'Protein'),
        ('disease', 'Disease'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query_type = models.CharField(max_length=10, choices=QUERY_TYPES)
    query_value = models.CharField(max_length=255)
    results = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']  # Order by most recently updated
    
    def __str__(self):
        return f"{self.user.username} - {self.query_type}: {self.query_value}"
