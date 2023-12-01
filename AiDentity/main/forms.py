from django import forms
from main.models import Face


class Faceform(forms.ModelForm):
    class Meta:
        model = Face
        fields = ["image"]
