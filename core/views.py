from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.contrib import messages
from django import forms
from django.core.validators import EmailValidator
from django.core.exceptions import ValidationError
import re
from .models import PredictionHistory
from .ml_model import DrugRepurposingModel

# Initialize the ML model
ml_model = DrugRepurposingModel()

# Custom Registration Form with Email and Strong Password
class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        validators=[EmailValidator(message="Enter a valid email address")],
        widget=forms.EmailInput(attrs={
            'placeholder': 'your.email@example.com',
            'autocomplete': 'email'
        })
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove class attribute that might interfere
        self.fields['username'].widget.attrs.update({
            'placeholder': 'Choose a username',
            'autocomplete': 'username'
        })
        self.fields['password1'].widget.attrs.update({
            'placeholder': 'Create a strong password',
            'autocomplete': 'new-password'
        })
        self.fields['password2'].widget.attrs.update({
            'placeholder': 'Confirm your password',
            'autocomplete': 'new-password'
        })
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("This email is already registered.")
        return email
    
    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        
        # Strong password validation
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter.")
        
        if not re.search(r'[a-z]', password):
            raise ValidationError("Password must contain at least one lowercase letter.")
        
        if not re.search(r'[0-9]', password):
            raise ValidationError("Password must contain at least one number.")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValidationError("Password must contain at least one special character (!@#$%^&*...).")
        
        return password
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to Drug Repurposing Platform.')
            return redirect('home')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = AuthenticationForm()
    
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def home_view(request):
    results = None
    query_type = None
    query_value = None
    
    if request.method == 'POST':
        query_type = request.POST.get('query_type')
        query_value = request.POST.get('query_value')
        
        if query_type and query_value:
            # Get predictions from ML model
            results = ml_model.predict(query_type, query_value)
            
            # Check if this exact query already exists for this user
            existing = PredictionHistory.objects.filter(
                user=request.user,
                query_type=query_type,
                query_value=query_value
            ).first()
            
            if existing:
                # Update the existing entry
                existing.results = results
                existing.save()  # This will update the updated_at timestamp
            else:
                # Create new entry
                PredictionHistory.objects.create(
                    user=request.user,
                    query_type=query_type,
                    query_value=query_value,
                    results=results
                )
    
    # Get recent history (ordered by most recently updated)
    recent_history = PredictionHistory.objects.filter(user=request.user).order_by('-updated_at')[:5]
    
    context = {
        'results': results,
        'query_type': query_type,
        'query_value': query_value,
        'recent_history': recent_history,
    }
    
    return render(request, 'home.html', context)

@login_required
@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard_view(request):
    # Get filter parameter
    view_filter = request.GET.get('view', None)
    
    total_users = User.objects.count()
    total_predictions = PredictionHistory.objects.count()
    
    # Statistics by query type
    stats = {
        'drug': PredictionHistory.objects.filter(query_type='drug').count(),
        'protein': PredictionHistory.objects.filter(query_type='protein').count(),
        'disease': PredictionHistory.objects.filter(query_type='disease').count(),
    }
    
    # Determine what to show based on filter
    if view_filter == 'users':
        # Show all users
        users_list = User.objects.all().order_by('-date_joined')
        context = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'stats': stats,
            'view_filter': 'users',
            'users_list': users_list,
        }
    elif view_filter in ['drug', 'protein', 'disease']:
        # Show predictions filtered by type
        filtered_predictions = PredictionHistory.objects.filter(query_type=view_filter).order_by('-updated_at')
        context = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'stats': stats,
            'view_filter': view_filter,
            'filtered_predictions': filtered_predictions,
        }
    elif view_filter == 'all_predictions':
        # Show all predictions
        all_predictions = PredictionHistory.objects.all().order_by('-updated_at')
        context = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'stats': stats,
            'view_filter': 'all_predictions',
            'all_predictions': all_predictions,
        }
    else:
        # Default view - show recent activity
        recent_predictions = PredictionHistory.objects.all().order_by('-updated_at')[:20]
        context = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'recent_predictions': recent_predictions,
            'stats': stats,
        }
    
    return render(request, 'admin_dashboard.html', context)
