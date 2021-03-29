from lib.functionality.torch_helper import load_set, get_default_device, to_device
from lib.functionality.data import load_params, preproccessing, unpack_img_dataset, push_to_folder
from lib.functionality.binance_helper import fetch_historic
from lib.functionality.model import *

import time

cwd = os.path.abspath(os.getcwd())
pointfp = f'{cwd}/lib/json/pointers.json'
fp, modelState = [load_params(pointfp, deploy=True)['paths'][key] for key in load_params(pointfp, deploy=True)['paths']]
data_params, cloud_params, _, _ = [load_params(fp)[key] for key in load_params(fp)]

classMap = {0: 'hold', 1: 'sell', 2: 'buy'}
dirName = 'CNN Deploy'

"""
Load the model
"""
model = to_device(ResNet9(3, 3), device)
model.load_state_dict(torch.load(modelState))

P = 1000
pos = 0
loop_outcome = {
    'Timestamp': [],
    'Prediction': [],
    'Position': [],
    'Funds': []
}

"""
Main deploy loop
"""

while True:
    
    ta_df = fetch_historic(data_params)

    #Read in new data (previous 250 mins @ frame = 0.5)
    transformed_ta_df = preproccessing(ta_df)
    last_img = transformed_ta_df[-1]
    
    """
    From here, the resulting must recent image needs to be saved as a .png to a directory. Following 
    this the image needs to be 
        - loaded/transformed (image folder) using a pytorch dataloader
        - utilised for prediction
        - shunted into a dump folder so that dataloader is effectivly loading a folder of size 1
    """

    img_folder = push_to_folder(cloud_params['targetLoc'], dirName, pixels = last_img)
    prediction_data = ImageFolder(f'{cloud_params["targetLoc"]}{dirName}', transform=tt.ToTensor())
    img, _ = prediction_data[0]
    
    prediction = predict_image(img, model, classMap)
    print(f'Predicted: {prediction}')
    
    if prediction == 'sell':
        if pos > 0:
            P = pos * ta_df.close[-1]
            pos -= pos
        print(f'funds = {P}')
        
    elif prediction == 'buy':
        if P > 0:
            pos = P/ta_df.close[-1]
            P -= P    
        print(f'funds = {P}')
        
    else:
        print(f'funds = {P}')
        pass
    
    loop_outcome['Timestamp'].append(dt.datetime.now())
    loop_outcome['Prediction'].append(prediction)
    loop_outcome['Position'].append(pos)
    loop_outcome['Funds'].append(P)

    push_to_folder(cloud_params['targetLoc'], dirName, destination = 'dump')

    time.sleep(60*5)

