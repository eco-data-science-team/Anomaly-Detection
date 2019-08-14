# Anomaly-Detection

## Requirements

With **Conda**

    $ conda create --name <env> --file requirements.txt

Where:
- `<env>` is the name of the environment name you wish to create


With **Pip**

    $ pip install -r requirements.txt
---
## Setup

1. Under `src/config` folder open `lstmconfig`
2. Under `[SETUP]` > `eco_tools_path` change *ECO TOOLS PATH HERE* to the absolute path to your eco-tools folder:
    
        /Users/mypc/Desktop/eco-tools

3. If you wish you can change various parameters for the `model` section such as the percentage of data you want the model to train on. 

---
## Using the Jupyter Notebook

1. Open up `jupyter notebook` using the command:
    
        $ jupyter notebook
2. Move to the directory in which the `Anomaly_Notebook.ipynb` is located and begin runnig the notebook. Further instructions are within the notebook itself.