from django import forms
from AppOne.models import Custom_user, Signature


class Custom_user_form(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    class Meta:
        model = Custom_user
        fields = ('username', 'email', 'password', 'photo')

class Signature_form(forms.ModelForm):
    class Meta:
        model = Signature
        fields = '__all__'