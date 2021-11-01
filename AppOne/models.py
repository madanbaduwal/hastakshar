from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.




class Custom_user(AbstractUser):
    photo = models.ImageField(upload_to='user_photo')
    email = models.EmailField(unique=True)



class Signature(models.Model):
    signature = models.ImageField(upload_to='user_signature')
    # user = models.ForeignKey(Custom_user, on_delete=models.CASCADE, null=True)




