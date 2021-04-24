import os, json
import datetime as dt
from deploy import Deploy
from train import Build

"""
Main Module
============

Runs the entire package. If there is no saved model state discoverable, data will be checked for in local 
directories, a relational database will be queried or data will be pulled from Binance's API, after which
a model will be trained.
"""

def main():
    current_date = dt.datetime.today().strftime('%m-%d')

    with open(f'./lib/json/pointers.json', 'r') as model_state:
        trained_model = json.load(model_state)['paths']['modelPath']
        
    model_state.close()

    if os.path.isfile(trained_model):
        deployment = Deploy()
        deployment.run()

    else:
        model_foundations = Build(current_date)
        model_foundations.run()

if __name__ == '__main__':
    main()