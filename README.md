# EQ-Web
EQ-Web is a web project used for seismology including magnitude estimation and so on. <br>
But in reality, EQ-Web can be extended to any deep learning or machine learning models. <br>

## Installation
EQ-Web is based on [Django](https://docs.djangoproject.com) and [Django REST framework](https://www.django-rest-framework.org/). You can obtain relevant tutorials from [Machine Learning with Django](https://www.deploymachinelearning.com/).<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.9`<br><br>
Then, there are some Python packages need to be installed<br>
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3
conda install pyg -c pyg
conda install h5py==2.10.0
conda install django
conda install djangorestframework
conda install markdown
conda install django-filter
```

## Dataset Preparation
The Dataset used in our paper can be downloaded from [https://github.com/smousavi05/STEAD](https://github.com/smousavi05/STEAD).

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/dataset_structure.png)<br>

## Program Description
Firstly please go to the directory where [manager.py](https://github.com/czw1296924847/EQ-Web/blob/main/manage.py) is located, and then do other actions.

### 1. Generating Django Model
```
python manager.py makemigrations
python manager.py migrate
```
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/generate_model.png)<br>
We have defined three Django Models, including<br>
- EndPoint
- MagModel
- MagStatus
- MagRequest

### 2. Starting Service
`python manager.py runserver`<br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/run_server.png)<br>

### 3. Accessing Web and Entering Root Page
For Linux system, you can access web pages by using <br>
```
xdg-open http://{host}:{port}
```
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/root_page.png)
In our example, host = 127.0.0.1 and port = 8000.

### 4. Training Model
Go to `http://127.0.0.1:8000/estimate/magnitude_estimator/train` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/train_before.png)<br>

and provide example JSON input:
```
{
    "network": "CREIME",
    "device": "cuda:1",
    "lr": 0.0005,
    "decay": 0.0005,
    "batch_size": 64,
    "epochs": 100,
    "train_ratio": 0.75,
    "m": 100,
    "sm_scale": ["ml"],
    "name": "chunk2",
    "root": "/home/chenziwei2021/standford_dataset",
    "re_ad": "/home/chenziwei2021/pyn/paper/EQGraphNet/web/estimate/static/result"
}
```
and click the $\text{\color{blue}{POST}}$ button, the model starts training <br>
After a period of time (possibly a long time, based on the amount of data and model parameters), you should see views like <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/train_after.png)<br>

You can view the model training process in the background <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/train_process.png)

### 5. Testing Model
After training some model, go to `http://127.0.0.1:8000/estimate/magnitude_estimator/test` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/test_before.png)<br>

and provide example JSON input:
```
{
    "network": "CREIME",
    "device": "cuda:1",
    "lr": 0.0005,
    "decay": 0.0005,
    "batch_size": 64,
    "epochs": 100,
    "train_ratio": 0.75,
    "m": 100,
    "sm_scale": ["ml"],
    "name": "chunk2",
    "root": "/home/chenziwei2021/standford_dataset",
    "re_ad": "/home/chenziwei2021/pyn/paper/EQGraphNet/web/estimate/static/result"
}
```
and click the $\text{\color{blue}{POST}}$ button, the model starts testing, and after a shorter period of time, you should see views like <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/test_after.png)<br>

You can view the model testing process in the background <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/test_process.png)<br>


## Problems and Solutions

### 1. Testing model without Training
If you have not trained the model before testing, you will <br>
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/not_train.png)<br>
where the `CREIME` haven't been trained.

## Extension
If you plan to make your own models and data, you can take the following steps <br>

### ?. Defining Model Structure
Define your own model structure in [net.py](https://github.com/czw1296924847/EQ-Web/blob/main/func/net.py) <br>

### ?. Registrying Your Model
Do not forget to registry your model in [wsgi.py](https://github.com/czw1296924847/EQ-Web/blob/main/web/wsgi.py) <br>
