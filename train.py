# !pip install gputil
# !pip install psutil
# !pip install humanize
# !pip install python-resources
# !pip install python-binance
# !pip install bta-lib

#Import all necessary libraries
from lib.functionality.torch_helper import load_set, get_default_device, to_device
from lib.functionality.data import load_params, preproccessing, unpack_img_dataset
from lib.functionality.binance_helper import fetch_historic
from lib.functionality.model import *
import os
import json

cwd = os.path.abspath(os.getcwd())

#Ingest parameters file
print('Initialising training...')
print('========================')
fp = f'{cwd}/lib/json/parameters.json' 
data_params, cloud_params, loading_params, model_params = [load_params(fp)[key] for key in load_params(fp)]
print('  - Structure parameters parsed')
print('')
#Retrieve data and run preprocessing/feature engineering
print('  - Gathering data', end='')
ta_df = fetch_historic(data_params)
print(' ... Complete')
print('  - Generating feature matrix', end='')
preproccessing(ta_df) #returns the dataframe, assign to nothing if memory conscious as csv is downloaded
print(' ... Complete')
#Unpack CSV data into png images and generate file structure
dataset = unpack_img_dataset(cloud_params)

print('  - Loading model', end='')
#Pointer to torch device
device = get_default_device()

#Load split data into pytorch
train_dl = load_set(loading_params['trainDL'], device, dataset)
valid_dl = load_set(loading_params['validDL'], device, dataset)

#Instantiate model
model = to_device(ResNet9(3, 3), device)

#Model training init
opt_func = torch.optim.Adam
print(' ... Complete')
#Begin training
print('  - Commencing model training loop. This may take awhile!')
history = [evaluate(model, valid_dl)]
history += fit(
        model_params,
        model, 
        train_dl, 
        valid_dl,
        opt_func=opt_func
)

# performance_visualiser(targetLabels, predsMade, history, cloud_params['classNames']) #Performance metrics vis

pointers = {
    'history': {
        'history': history
    }, 
    'paths': {
        'paramPath': f'{cwd}/lib/json/parameters.json', 
        'modelPath': f'{cloud_params["targetLoc"]}cnn_model.pth'
    }
}

with open(f'{cwd}/lib/json/pointers.json', 'w') as outfile:
    json.dump(pointers, outfile)

torch.save(model.state_dict(), pointers['paths']['modelPath'])
print('========================')
print('Training complete. Model state saved to /lib/saves')