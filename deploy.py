import lib.functionality as fn
import lib.storage as st

import time, os, torch
import datetime as dt

import torchvision.transforms as tt
from torchvision.datasets import ImageFolder

class Deploy:
    """
    Deploy Module
    --------------

    Wrapper for model deployment. Holds a looping method for continuous deployment of a trained model.
    Out of sample predictions are read via Binance's API and reduced to the smallest possible dataframe
    for valid feature computation as a single image. The inference is conducted on single-image dataset 
    with the prediction informing a decision to buy, sell or hold the asset.
    """
    def __init__(self):
        """ 
        Takes an access date for current build date. Date is used to check whether data is available, or
        to push a download request to Binance. Generates a number of attributes for later computation.
        """

        pointfp = os.path.join(
            os.path.abspath(os.getcwd()),
            '/lib/json/pointers.json'
        )

        path_map = fn.processing.load_params(pointfp, deploy=True)['paths']

        fp, modelState = [
            path_map[key] for key in path_map
        ]

        self.data_params, self.cloud_params, _, _ = [
            fn.processing.load_params(fp)[key] for key in fn.processing.load_params(fp)[key]
        ]
        
        self.model = fn.device.to_device(
            fn.model.ResNet9(3, 3), 
            fn.device.get_default_device()
        )

        self.model.load_state_dict(
            torch.load(modelState)
        )

        self.P = 1000
        self.pos = 0

        self.loop_outcome = {
            'Timestamp': [],
            'Prediction': [],
            'Position': [],
            'Funds': []
        }

        self.classMap = {
            0: 'hold', 
            1: 'sell', 
            2: 'buy'
        }

    def run(self):
        """ 
        Main method for deploying the model. Handles new data ingest, processing and prediction. Additional 
        (dummy) functionality in that buy/sell actions are performed by the deployment module based on 
        predictions
        """
        while True:

            self.data = fn.data.fetch_historic(self.data_params)
            self.processed_data = fn.processing.tech_ind_features(self.data)

            last_img = self.processed_data[-1]

            st.pipeline.DirTools.push_to_folder(
                self.cloud_params['targetLoc'],
                self.data_params["ticker"], 
                pixels = last_img
            )

            prediction_data = ImageFolder(
                f'{self.cloud_params["targetLoc"]}/{self.data_params["ticker"]}', 
                transform=tt.ToTensor()
            )

            img, _ = prediction_data[0]
            
            prediction = fn.device.predict_image(
                img, 
                self.model, 
                self.classMap, 
                fn.device.get_default_device()
            )
            print(f'Predicted: {prediction}')
            
            if prediction == 'sell':
                if self.pos > 0:
                    self.P = self.pos * self.data.close[-1]
                    self.pos -= self.pos
                print(f'funds = {self.P}')
                
            elif prediction == 'buy':
                if self.P > 0:
                    self.pos = self.P/self.data.close[-1]
                    self.P -= self.P    
                print(f'funds = {self.P}')
                
            else:
                print(f'funds = {self.P}')
                pass
            
            self.loop_outcome['Timestamp'].append(dt.datetime.now())
            self.loop_outcome['Prediction'].append(prediction)
            self.loop_outcome['Position'].append(self.pos)
            self.loop_outcome['Funds'].append(self.P)

            st.pipeline.DirTools.push_to_folder(
                self.cloud_params['targetLoc'],
                self.data_params["ticker"], 
                destination = 'dump'
            )

            time.sleep(60*5)

