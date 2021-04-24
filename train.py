import lib.functionality as fn
import lib.storage as st
import os, json, torch
import datetime as dt
import pandas as pd

class Build:
    """
    Build Module
    --------------

    Wrapper for all processes involved in retrieving or downloading data from either source or 
    relational database. Wraps parameter and credential parsing, feature extraction, dataset 
    structuring and model training. Consol prinouts enable a less black-box'y UI.
    """
    def __init__(self, access_date):
        """ 
        Takes an access date for current build date. Date is used to check whether data is available, or
        to push a download request to Binance. Generates a number of attributes for later computation.
            * :param access_date(datetime.datetime): Current date
        """
        print('Initialising training...\n========================')

        self.cwd, self.access_date = os.path.abspath(os.getcwd()), access_date
        self.parameters = fn.processing.load_params(f'{self.cwd}/lib/json/parameters.json')

        self.data_params, self.cloud_params, self.loading_params, self.model_params = [
            self.parameters[key] for key in self.parameters
        ]

        self.raw_fp = f'./lib/datasets/raw/{self.data_params["ticker"]}{self.access_date}.csv'
        self.proc_fp = f'./lib/datasets/processed/{self.data_params["ticker"]}{self.access_date}.csv'
        self.device = fn.device.get_default_device()

        print('  - Structure parameters parsed\n')

    def _gather(self):
        """ 
        Called by run method. Handles data requests from relational database or Binance. Checks for 
        data from current day by searching for files with format '[ticker][current date].csv' and 
        pulls from Binance API, uploads to relational database and forward-passes to feature generation 
        processes. 
        
        All variables are either computed or inherited from the instantiated Build
        """
        print('  - Gathering data', end='')

        if not st.pipeline.DirTools.file_found(self.access_date, self.data_params['ticker']):
            self.credfilepath = os.path.join(
                "./lib/json/_credentials/", 
                "classifierDB.json"
            )

            st.sql.SQLTools.parse_and_upload(
                self.credfilepath, 
                fn.data.fetch_historic(self.data_params)
            )
        
            conn = st.sql.SQLTools.pgconnect(self.credfilepath)
            self.data = pd.read_sql(
                f'SELECT * from {self.data_params["ticker"]}', 
                conn
            )
            conn.close

            self.data.to_csv(
                self.raw_fp, 
                index = False
            )

        else:
            self.data = pd.read_csv(
                self.raw_fp
            )

        print(' ... Complete')

    def _process(self):
        """ 
        Called by run method. Processes raw data and generates features. 
        Handles data requests from relational database or Binance. Checks for data from current day
        by searching for files with format '[ticker][current date].csv' and pulls from Binance API,
        uploads to relational database and forward-passes to feature generation processes. 
        """
        print('  - Generating feature matrix', end='')

        self.prepared_data = fn.processing.tech_ind_features(self.data)

        pd.DataFrame(self.prepared_data).to_csv(
            self.proc_fp, 
            index = False
        )
        print(' ... Complete')

        dataset = fn.processing.unpack_img_dataset(
            self.cloud_params,
            self.data_params["ticker"],
            self.proc_fp
        )

        print('  - Loading model', end='')
        self.train_dl = fn.device.load_set(
            self.loading_params['trainDL'], 
            self.device, 
            dataset
        )
        self.valid_dl = fn.device.load_set(
            self.loading_params['validDL'], 
            self.device, 
            dataset
        )

        self.model = fn.device.to_device(fn.model.ResNet9(3, 3), self.device)

        print(' ... Complete')

    def run(self):
        """ 
        Main class method. Calls internal methods to access and process ticker data from Binace API
        into a .png image dataset. A ResNet9 model is trained on the generated data to predict 3 classes
        "hold", "buy", and "sell". The trained model-state is saved to ./lib/saves/ for deployment and 
        out of sample prediction or constant runtime applications.
        """
        self._gather()
        self._process()
        
        print('  - Commencing model training loop. This may take awhile!')

        history = [
            fn.model.evaluate(
                self.model, 
                self.valid_dl
            )
        ]

        history += fn.model.fit(
                self.model_params,
                self.model, 
                self.train_dl, 
                self.valid_dl,
                optimiser_f=torch.optim.Adam
        )

        pointers = {
            'history': {
                'history': history
            }, 
            'paths': {
                'paramPath': f'{self.cwd}/lib/json/parameters.json', 
                'modelPath': f'./lib/saves/{self.data_params["ticker"]}{self.access_date}cnn_model.pth'
            }
        }

        with open(f'{self.cwd}/lib/json/pointers.json', 'w') as outfile:
            json.dump(pointers, outfile)

        torch.save(self.model.state_dict(), pointers['paths']['modelPath'])

        print('========================')
        print('Training complete. Model state saved at ./lib/saves')