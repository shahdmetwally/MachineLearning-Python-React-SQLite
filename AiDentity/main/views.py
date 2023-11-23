from django.shortcuts import render, HttpResponse

def home(request):
    return render(request, "base.html")

# Create your views here.
