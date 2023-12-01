from django.shortcuts import render
from main.forms import Faceform
from main.utils import load_trained_model

# Create your views here.


def home(request):
    form = Faceform()

    if request.method == "POST":
        form = Faceform(request.POST or None, request.FILES or None)
        if form.is_valid():
            form.save(commit=True)

    return render(request, "home.html", {"form": form})


def history(request):
    return render(request, "history.html")


def about(request):
    return render(request, "about.html")
