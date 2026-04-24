from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'query_type', 'query_value', 'created_at']
    list_filter = ['query_type', 'created_at']
    search_fields = ['user__username', 'query_value']
    readonly_fields = ['created_at']
