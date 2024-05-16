# IMLO Coursework
Repository for University of York, IMLO coursework: flower classification from scratch, using the flowers-102 dataset.
## Installation
Create a new virtual environment.
```shell
python -m venv .venv
````
activate on POSIX shell.
```
source .venv/bin/activate
```
activate on Windows, for cmd.exe.
```shell
.venv\Scripts\activate.bat
```
activate on Windows, for PowerShell.
```
.venv\Scripts\Activate.ps1
```
Install dependencies:
```shell
pip install -r requirements.txt
```
## Usage
To train the model, simply run train.py:
```shell
python train.py
```
If you recieve an error relating to multithreading in some way, please change num_workers in
imlo-coursework/load_data.py

This will create a file called `model.pt`, which is then loaded when testing.

To test the model, simply run test.py:
```shell
python test.py
```
It will output a loss and an accuracy for the model. This varies between about 70%-73%, due to randomness when
training.

To calculate the normal values, run:
```shell
python main.py
```
This will calculate the values used when normalising the datasets. This is calculated using only the training data.

