import pandas as pd
import numpy as np
import datetime as dt
import time, csv, os, itertools, shutil, json, random
import btalib

from itertools import compress
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

def bool_convert(s):
    """
    Parse string booleans from parameters
        * :param s(str): String to convert
    :return s(bool): Boolean type
    """
    try:
        if s == "True":
            return True
        elif s == "False":
            return False
        else:
            return s
    except TypeError:
        return s

def int_convert(s):
    """
    Parse string int from parameters
        * :param s(str): String to convert
    :return s(int): Int type
    """
    try:
        return int(s)
    except ValueError:
        return s
    except TypeError:
        return s
                
def load_params(filePath, deploy = False):
    """
    Parse parameters json
        * :param filePath(str): Location of parameters file
        * :param deploy(bool): Denotation for whether parameters are being loaded by deployment code

    :return params(dict): Python dictionary of parameters with correct dtypes
    """
    with open(filePath) as f:
        params = json.load(f)
    if deploy:
        return params
    else: 
        for split in ['validDL', 'trainDL']:
            params['loadingParams'][split]['shuffle'] = bool_convert(params['loadingParams'][split]['shuffle'])

        for key in params:
            for subkey in params[key]:
                if subkey == 'shuffle':
                    params[key][subkey] = bool_convert(params[key][subkey])
                else:
                    params[key][subkey] = int_convert(params[key][subkey])
        
    f.close()
    return params

def unpack_img_dataset(params, directory, file_name):
    """
    Unpack an image dataset stored as raw data (csv or otherwise) into .png's. Creates file structure for pytorch loading
    and handles train/test/validation splitting internally
        * :param params(dict): Location of parameters file
        * :param directory(str): Name of directory to check for, or create to store images
        * :param file_name(str): Name of .csv file of featureset for image creation   

    :return parentPath(str): Path to parent directory of the dataset
    """
    _, targetDir, classNames, imSize = [value for key, value in params.items()]
    counter = {}
    labelMap = {}
    filePathMap = {
        0:{}, 
        1:{}
    }
    classFilePaths = {
        'train':[], 
        'test':[]
    }
    
    for i, j in zip(range(0,len(classNames)), classNames):
        labelMap[str(i)] = j
        filePathMap[0][str(i)] = ''
        filePathMap[1][str(i)] = ''

    #Paths for the directory
    parentPath = os.path.join(targetDir, directory)
    trainPath = os.path.join(parentPath, 'train')
    testPath = os.path.join(parentPath, 'test')
    try:
        os.mkdir(parentPath)
        os.mkdir(trainPath)
        os.mkdir(testPath)
        print(f'Directory \'{directory}\' created')
        for elem in classNames:
            fpTrain = os.path.join(trainPath, elem)
            fpTest = os.path.join(testPath, elem)
            classFilePaths['train'].append(fpTrain)
            classFilePaths['test'].append(fpTest)
            os.mkdir(fpTrain)
            os.mkdir(fpTest)
            print(f'    {elem} class train/test directories created')
            
        for i, itemTrain, itemTest in zip(range(len(classNames)), classFilePaths['train'], classFilePaths['test']):
            i = str(i)
            filePathMap[0][i] = itemTrain
            filePathMap[1][i] = itemTest

    except FileExistsError:
        print(f'{directory} already exists - consider deleting the directory for a clean install!')
    
    numSamples = len(pd.read_csv(file_name))
    test_idx = [random.randint(0, numSamples) for i in range(0, int(numSamples * 0.2))]
    print(f'Unpacking {file_name}...')
    print('Please wait...')
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file)

        next(csv_reader)
        fileCount = 0
        for row in csv_reader:
            
            if fileCount % 1000 == 0:
                print(f'Unpacking {fileCount}/{numSamples}...', end = ' ')

            pixels = row[:-1] # without label
            pixels = np.array(pixels, dtype='float64')
            pixels = pixels.reshape((imSize, imSize, 3))
            image = Image.fromarray(pixels, 'RGB')

            label = row[-1][0]

            if label not in counter: counter[label] = 0
            counter[label] += 1

            filename = f'{labelMap[label]}{counter[label]}.png'

            if fileCount in test_idx:
                filepath = os.path.join(filePathMap[1][label], filename)

            else:
                filepath = os.path.join(filePathMap[0][label], filename)

            image.save(filepath)
            
            if (fileCount % 999 == 0) and (fileCount != 9): print(f'Completed')
            fileCount += 1

        print(f'Unpacking complete. {fileCount} images parsed.')
    
    return parentPath

def tech_ind_features(data):
    """
    Generate technical indicators 
        * :param data(pd.DataFrame): Raw data for processing
       
    :return transformed_df(pd.DataFrame): Dataframe of features, sample balanced and normalised
    """
    data['smoothed_close'] = data.close.rolling(9).mean().rolling(21).mean().shift(-15)
    data['dx'] = np.diff(data['smoothed_close'], prepend=data['smoothed_close'][0])
    data['dx_signal'] = pd.Series(data['dx']).rolling(9).mean()
    data['ddx'] = np.diff(np.diff(data['smoothed_close']), prepend=data['smoothed_close'][0:2])

    data['labels'] = np.zeros(len(data))
    data['labels'].iloc[[(data.ddx < 0.1) & (data.dx <= 0) & (data.dx_signal > 0)]] = 1
    data['labels'].iloc[[(data.ddx > -0.075) & (data.dx >= 0) & (data.dx_signal < 0)]] = 2

    #Filter and drop all columns except close price, volume and date (for indexing)
    relevant_cols = list(
                        compress(
                            data.columns,
                            [False if i in [1, 3, 4, 5, 6, len(data.columns)-1] else True for i in range(len(data.columns))]
                        )
                    )

    data = data.drop(columns=relevant_cols).rename(columns={'open_date_time': 'date'})
    data.set_index('date', inplace = True)

    #Define relevant periods for lookback/feature engineering
    periods = [
        9, 14, 21, 
        30, 45, 60,
        90, 100, 120
    ]

    #Construct technical features for image synthesis
    for period in periods:
        data[f'ema_{period}'] = btalib.ema(
                                    data.close,
                                    period = period
                                ).df['ema']
        data[f'ema_{period}_dx'] = np.append(np.nan, np.diff(btalib.ema(
                                                                data.close,
                                                                period = period
                                                            ).df['ema']))
        data[f'rsi_{period}'] = btalib.rsi(
                                    data.close,
                                    period = period
                                ).df['rsi']
        data[f'cci_{period}'] = btalib.cci(
                                        data.high,
                                        data.low,
                                        data.close,
                                        period = period
                                ).df['cci']
        data[f'macd_{period}'] = btalib.macd(
                                    data.close,
                                    pfast = period,
                                    pslow = period*2,
                                    psignal = int(period/3)
                                ).df['macd']
        data[f'signal_{period}'] = btalib.macd(
                                        data.close,
                                        pfast = period,
                                        pslow = period*2,
                                        psignal = int(period/3)
                                    ).df['signal']
        data[f'hist_{period}'] = btalib.macd(
                                    data.close,
                                    pfast = period,
                                    pslow = period*2,
                                    psignal = int(period/3)
                                ).df['histogram']
        data[f'volume_{period}'] = btalib.sma(
                                        data.volume,
                                        period = period
                                    ).df['sma']
        data[f'change_{period}'] = data.close.pct_change(periods = period)

    data = data.drop(data.query('labels == 0').sample(frac=.90).index)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    data_trimmed = data.loc[:,  'ema_9':]
    data_trimmed = pd.concat(
                        [data_trimmed, 
                        data_trimmed.shift(1), 
                        data_trimmed.shift(2)],
                        axis = 1
                    )

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    transformed_data = mm_scaler.fit_transform(data_trimmed[24:])
    transformed_data = np.c_[
        transformed_data, 
        pd.to_numeric(
            data.labels[24:],
            downcast = 'signed'
        ).to_list()
    ]

    return transformed_data