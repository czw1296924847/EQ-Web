# EQ-Web
EQ-Web is a web project used for seismology including magnitude estimation and so on. <br>
But in reality, EQ-Web can be extended to any deep learning or machine learning models. <br>

## Installation
EQ-Web is based on [Django](https://docs.djangoproject.com) and [Django REST framework](https://www.django-rest-framework.org/). You can obtain relevant tutorials from [Machine Learning with Django](https://www.deploymachinelearning.com/).<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.9`<br><br>
Then, there are some Python packages need to be installed<br>
`conda install pytorch torchvision torchaudio cudatoolkit=11.3`<br>
`conda install pyg -c pyg`<br>
`conda install h5py==2.10.0`<br>
`conda install django`<br>
`conda install djangorestframework`<br>
`conda install markdown`<br>
`conda install django-filter`<br>

## Dataset Preparation
The Dataset used in our paper can be downloaded from [https://github.com/smousavi05/STEAD](https://github.com/smousavi05/STEAD).

![image](https://github.com/czw1296924847/MagInfoNet/blob/main/image/dataset_structure.png)

## Program Description
Firstly please go to the directory where [manager.py](https://github.com/czw1296924847/EQ-Web/blob/main/manage.py) is located, and then do other actions.

### Generating Django Model
`python manager.py makemigrations`<br>
`python manager.py migrate`<br>

### Starting Service
`python manager.py runserver`<br>
