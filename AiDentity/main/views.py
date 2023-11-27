from django.shortcuts import render

# Create your views here.


def home(request):
    return render(request, "home.html")


def history(request):
    return render(request, "history.html")


def about(request):
    return render(request, "about.html")
