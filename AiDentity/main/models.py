from django.db import models

# Create your models here.


class Face(models.Model):
    id = models.AutoField(primary_key=True)
    # target = models.IntegerField()  # integer
    # name = models.TextField(default="face")  # text not null
    image = models.ImageField(upload_to="images/")  # blob not null

    def __str__(self):
        return str(self.id)
