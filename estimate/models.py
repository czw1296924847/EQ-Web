from django.db import models


class User(models.Model):
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=100)

    def __str__(self):
        return self.username


class DlModel(models.Model):
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1000)
    version = models.CharField(max_length=128, default="0.0.1")
    owner = models.CharField(max_length=128)
    created_at = models.DateField(auto_now_add=True)
    situation = models.CharField(max_length=128, default="Free")
    path_data = models.CharField(max_length=128, blank=True)
    library = models.CharField(max_length=5000, blank=True)
    code_data = models.CharField(max_length=50000, blank=True)
    code_model = models.CharField(max_length=50000, blank=True)
    code_train = models.CharField(max_length=50000, blank=True)
    code_test = models.CharField(max_length=50000, blank=True)
    code_run = models.CharField(max_length=50000, blank=True)

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self._meta.fields}

    def __str__(self):
        return self.name


class DlModelStatus(models.Model):
    name = models.CharField(max_length=128)
    process = models.CharField(max_length=50000)

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self._meta.fields}


class Feature(models.Model):
    param = models.CharField(max_length=128)
    description = models.CharField(max_length=1000)
