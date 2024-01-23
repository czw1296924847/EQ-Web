# EQ-Web-BackEnd
EQ-Web-BackEnd is a web backend used for seismology including magnitude estimation and so on. <br>
But in reality, EQ-Web-BackEnd can be extended to any deep learning or machine learning models. <br>

This can serve as the backend framework for [EQ-Web-FrontEnd]()<br>

## Installation
EQ-Web-BackEnd is based on [Django](https://docs.djangoproject.com) and [Django REST framework](https://www.django-rest-framework.org/). You can obtain relevant tutorials from [Machine Learning with Django](https://www.deploymachinelearning.com/).<br>
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
<a name="section-begin"></a> Firstly please go to the directory where [manager.py](https://github.com/czw1296924847/EQ-Web/blob/main/manage.py) is located, and then do other actions.

### 1. Generating Django Model
```
python manager.py makemigrations
python manager.py migrate
```
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/generate_model.png)<br>
We have defined three Django Models, including<br>
- **EndPoint**:<br>
API endpoints,       including `name`, `owner`, `created_at` <br>
<!---
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/EndPoint.png)<br>
-->

- **MagModel**:<br>
Model information,   including `name`, `description`, `code`, `version`, `owner`, `created_at`, `parent_endpoints` <br>
<!---
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagModel.png)<br>
-->

- **MagStatus**:<br>
Model status,       including `status`, `active`, `created_by`, `parent_model` <br>
<!---
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagStatus.png)<br>


- **MagRequest**:<br>
Requests to model,   including `input_data`, `full_response`, `response`, `feedback`, `created_by`, `parent_model`
<!---
![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/MagRequest.png)<br>

-->

### 2. Starting Service
To start the web server, you should use
```
python manager.py runserver 0.0.0.0 8000
```
then in backgroudn system, you can see <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/run_server.png)<br>

### 3. Accessing Web and Entering Root Page
For Linux system, you can access web pages by using <br>
```
xdg-open http://{host}:{port}/estimate
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
    "network": "EQGraphNet",
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
    "network": "EQGraphNet",
    "device": "cuda:1",
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

### 6. Explain Meaning of Parameters
```
network:      the used model, including "EQGraphNet", "MagInfoNet", "MagNet", "ConvNetQuakeINGV", "CREIME"
device:       the device of Pytorch model, including "cpu", "cuda:0", "cuda:1"
lr:           the learning rate of model training, default 0.0005
decay:        the weight decay of model training, default 0.0005
batch_size:   batch number for Dataloader, default 64
train_ratio:  ration of training set, default 0.75
m:            the number of data set, defalut 200000
sm_scale:     magnitude scale, including ["ml"], ["md"], ["ml", "md"]
name:         the name of STEAD chunk, including "chunk2", "chunk3", "chunk4"
root:         the directory of data set, defalut "/home/chenziwei2021/standford_dataset"
re_ad:        the directory of saved model, default "/home/chenziwei2021/pyn/paper/EQGraphNet/web/estimate/static/result"
```

## Problems and Solutions

### 0. Universal method
There are two **violent** methods that can solve most bugs. So you should use them with caution.<br>

### 0.1 Extremely Violent
Remove the web project. For Linux system, firstly make sure you have backed up all files (excluding temporary files), then
```
cd ..
rm -rf web
```
Upload files to background system, and start over from the first step in [Program Description](#section-begin)<br>

### 0.2 Not so violent
Remove the db.sqlite3 file. For Linux system, you can
```
rm db.sqlite3
```
Re-execute the command to generate Django database, then restart the server
```
python manager migrate
python manager runserver
```


### 1. Testing model without Training
If you have not trained the model before testing, you will <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/not_train.png)<br>

where the `CREIME` haven't been trained.


## Extension
If you plan to make your own models and data, you can take the following steps <br>

### E.1. Defining Your Model Structure
Define your own model structure in [net.py](https://github.com/czw1296924847/EQ-Web/blob/main/func/net.py) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/net.png)<br>

### E.2. Serving Your Model
Implement your model code in [network.py](https://github.com/czw1296924847/EQ-Web/blob/main/estimate/network.py) <br>
EQ-Web provide a class by defining: `pre_train`, `training`, `pre_test`, `testing`. Of course, you can choose better approaches. <br>

- **Wait**: Before modifying, please confirm all of used parameters. For example, if you don't want to know the directory of model and data, fix their values and retrieve them from `pre_train` and `pre_test` <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/init.png)<br>


- **pre_train**: Obtain parameters required before training. This is related to the format of input JSON to [4. Training Model](#section-train) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/pre_train.png)<br>


- **training**: Training model, mainly including: loading data, iterative training, updating weights. <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/training.png)<br>


- **pre_test**: Obtain parameters required before testing. This is related to the format of input JSON to [5. Testing Model](#section-test) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/pre_test.png)<br>


- **testing**: Testing model, mainly including: calculating metrics.<br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/testing.png)<br>


### E.3. Registrying Your Model
Do not forget to registry your model in [wsgi.py](https://github.com/czw1296924847/EQ-Web/blob/main/web/wsgi.py) <br>

![image](https://github.com/czw1296924847/EQ-Web/blob/main/image/wsgi.png)<br>


### E.r. Restarting Server
Use `Ctrl C` to stop the running server, and then restart it <br>
```
python manager.py runserver 0.0.0.0 8000
```

## API Reference
Here we provide technical reference for APIs. They will describe the usage of them <br>

### func/process.py

- **get_train_or_test_idx**: Randomly generate indexes for training and testing sets<br>
`num`: the sample number of training and testing sets<br>
`num_train`: the sample number of training set<br>

- **SelfData**: Custom dataset, used for model training in deep learning <br>
`data`: input data, in our example is earthquake signals<br>
`label`: input label, in our example is magnitude <br>
`*args`: other input information which may be used during model training and testing <br>

- **remain_sm_scale**: only retain earthquake samples of specific magnitude scales <br>
`data`: earthquake signals <br>
`df`: earthquake information <br>
`label`: magnitude value <br>
`scale`: magnitude scale <br>

- **get_item_by_dim**: Determine the index based on dimension, where the index is used in `SelfData.__getitem__` <br>
`data`: input data <br>
`item`: dataset item, this is given by `SelfData.__getitem__` <br>

### func/net.py

- **cal_rmse_one_arr**: calculate the root mean square error <br>
`true`: the true label <br>
`pred`: the predicted result by models <br>

- **cal_r2_one_arr**: calculate the coefficient of determination <br>
`true`: the true label <br>
`pred`: the predicted result by models <br>

- **get_edge_info**: get the edge-index and edge-weight used for GNN ([torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)) layers <br>
`k`: the number of ajancent nodes of 'ts_un' and 'tg' <br>
`num_nodes`: the number of global graph structure <br>
`adm_style`: the style of adjacency matrix, including 'ts_un' and 'tg' <br>
`device`: the device of Pytorch model, including 'cpu', 'cuda:1' and 'cuda:2' <br>

- **get_gnn**: get GNN layer ([torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)) <br>
`gnn_style`: the GNN style, including 'gcn', 'cheb', 'unimp' and so on <br>
`in_dim`: the input dimension <br>
`out_dim`: the output dimension <br>

- **tran_adm_to_edge_index**: transport adjacency matrix to the edge-index <br>
`adm`: the adjacency matrix <br>

### estimate/views.py

- **ModelTrainView**: View of web page for model training <br>
`post`: when you click the $\text{\color{blue}{POST}}$ button, go to this function and start model training
  
- **ModelTestView**: View of web page for model testing <br>
`post`: when you click the $\text{\color{blue}{POST}}$ button, go to this function and start model testing

```
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash
getconf GNU_LIBC_VERSION
npx nvm install v16.20.2
```
In our example, the version of glibc is 2.27, so we download v16.20.2, and please add `npx` if you are not root.
