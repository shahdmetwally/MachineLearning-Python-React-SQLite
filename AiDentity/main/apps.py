from django.apps import AppConfig

# from .utils import load_trained_model


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "main"

    # def ready(self):
    # Load the model when the app is ready
    # self.model = load_trained_model()
