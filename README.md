# EQ-Web
EQ-Web is a web project used for seismology including magnitude estimation and so on. <br>
But in reality, EQ-Web can be extended to any deep learning or machine learning models. <br>


## Installation
EQ-Web is based on [Django](https://docs.djangoproject.com) and [Django REST framework](https://www.django-rest-framework.org/). You can obtain relevant tutorials from [Machine Learning with Django](https://www.deploymachinelearning.com/).<br>
Firstly please create a virtual environment for yourself<br>
```
conda create -n your-env-name python=3.9
```
Then, there are some Python packages need to be installed<br>
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3
conda install pyg -c pyg
conda install h5py==2.10.0
conda install django
conda install djangorestframework
conda install markdown
conda install django-filter
conda install tqdm
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
- **EndPoint**:<br>
API endpoints,       including `name`, `owner`, `created_at` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/EndPoint.png)<br>


- **MagModel**:<br>
Model information,   including `name`, `description`, `code`, `version`, `owner`, `created_at`, `parent_endpoints` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagModel.png)<br>


- **MagStatus**:<br>
Model status,       including `status`, `active`, `created_by`, `parent_model` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagStatus.png)<br>


- **MagRequest**:<br>
Requests to model,   including `input_data`, `full_response`, `response`, `feedback`, `created_by`, `parent_model`

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagRequest.png)<br>



### 2. Starting Service
To start the web server, you should use
```
python manager.py runserver
```
then in backgroudn system, you can see <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/run_server.png)<br>

### 3. Accessing Web and Entering Root Page
For Linux system, you can access web pages by using <br>
```
xdg-open http://{host}:{port}
```
then you will see <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/root_page.png)<br>

In our example, host = 127.0.0.1 and port = 8000.


### 4. Training Model
<a name="section-train"></a>  Go to `http://127.0.0.1:8000/estimate/magnitude_estimator/train` <br>

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

After a period of time (maybe a long time for model training), you should see <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/train_after.png)<br>

You can view the model training process in background system <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/train_process.png)

### 5. Testing Model
<a name="section-test"></a> After training some model, go to `http://127.0.0.1:8000/estimate/magnitude_estimator/test` <br>

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
and click the $\text{\color{blue}{POST}}$ button, the model starts testing <br>

After a shorter period of time, you should see views like <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/test_after.png)<br>

You can view the model testing process in background system <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/test_process.png)<br>


## Problems and Solutions

### 1. Testing model without Training
If you have not trained the model before testing, you will <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/not_train.png)<br>

where the `CREIME` haven't been trained.


## Extension
If you plan to make your own models and data, you can take the following steps <br>

### 1. Defining Your Model Structure
Define your own model structure in [net.py](https://github.com/czw1296924847/EQ-Web/blob/main/func/net.py) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/net.png)<br>

### 2. Serving Your Model
Implement your model code in [network.py](https://github.com/czw1296924847/EQ-Web/blob/main/estimate/network.py) <br>
EQ-Web provide a class by defining: `pre_train`, `training`, `pre_test`, `testing`. Of course, you can choose better approaches. <br>

- **pre_train**: Obtain parameters required before training. This is related to the format of input JSON to [4. Training Model](#section-train) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/pre_train.png)<br>


- **training**: Training model, mainly including: loading data, iterative training, updating weights. <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/training.png)<br>


- **pre_test**: Obtain parameters required before testing. This is related to the format of input JSON to [5. Testing Model](#section-test) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/pre_test.png)<br>


- **testing**: Testing model, mainly including: calculating metrics.<br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/testing.png)<br>


### 3. Registrying Your Model
Do not forget to registry your model in [wsgi.py](https://github.com/czw1296924847/EQ-Web/blob/main/web/wsgi.py) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/wsgi.png)<br>


## API

