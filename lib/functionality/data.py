import pandas as pd
import numpy as np
import datetime as dt
import time, csv, os, itertools, shutil, json, random
import btalib

from itertools import compress
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

def bool_convert(s):
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
    try:
        return int(s)
    except ValueError:
        return s
    except TypeError:
        return s
                
def load_params(filePath, deploy = False):
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

def unpack_img_dataset(params):
    _, targetDir, dirName, file, classNames, imSize = [value for key, value in params.items()]
    counter = {}
    labelMap = {}
    filePathMap = {0:{}, 1:{}}
    classFilePaths = {'train':[], 'test':[]}
    #generates a dict of each numerical label with its textual class name
    
    for i, j in zip(range(0,len(classNames)), classNames):
        labelMap[str(i)] = j
        filePathMap[0][str(i)] = ''
        filePathMap[1][str(i)] = ''

    #Paths for the directory
    parentPath = os.path.join(targetDir, dirName)
    trainPath = os.path.join(parentPath, 'train')
    testPath = os.path.join(parentPath, 'test')
    try:
        os.mkdir(parentPath)
        os.mkdir(trainPath)
        os.mkdir(testPath)
        print(f'Directory \'{dirName}\' created')
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
        print(f'{dirName} already exists - consider deleting the directory for a clean install!')
    
    numSamples = len(pd.read_csv(file))
    test_idx = [random.randint(0, numSamples) for i in range(0, int(numSamples * 0.2))]
    print(f'Unpacking {file}...')
    print('Please wait...')
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)

        # skip headers
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

            if label not in counter:
                counter[label] = 0
            counter[label] += 1

            filename = f'{labelMap[label]}{counter[label]}.png'

            if fileCount in test_idx:
                filepath = os.path.join(filePathMap[1][label], filename)

            else:
                filepath = os.path.join(filePathMap[0][label], filename)

            image.save(filepath)
            
            if (fileCount % 999 == 0) and (fileCount != 9):
                print(f'Completed')
            fileCount += 1
        print(f'Unpacking complete. {fileCount} images parsed.')
    
    return parentPath

def push_to_folder(targetDir, dirName, pixels=None, imSize=9, destination='current'):
    #Paths for the directory
    parentPath = os.path.join(targetDir, dirName)
    currentPath = os.path.join(parentPath, 'current')
    dumpPath = os.path.join(parentPath, 'dump')
    try:
        os.mkdir(parentPath)
        os.mkdir(currentPath)
        os.mkdir(dumpPath)

        print(f'Directory \'{dirName}\' created')

    except FileExistsError:
        print(f'{dirName} already exists - pushing image to {currentPath}')

    if destination == 'current':
        filename = f'prediction.png'
        filepath = os.path.join(currentPath, filename)

        pixels = pixels.reshape((imSize, imSize))
        image = Image.fromarray(pixels, 'L')

        image.save(filepath)
        print(f'Image saved to {currentPath}')

    else:
        num_in_dir = len(os.listdir(dumpPath))
        filename = f'prection{num_in_dir + 1}.png'
        filepath = os.path.join(dumpPath, filename)

        shutil.move(currentPath+'/prediction.png', filepath)
        print(f'Image moved to {dumpPath}')

    return currentPath


def preproccessing(data):

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

    #Coerce to correct datatype
    data.volume = pd.to_numeric(data.volume)
    data.close = pd.to_numeric(data.close)
    data.high = pd.to_numeric(data.high)
    data.low = pd.to_numeric(data.low)

    #Define relevant periods for lookback/feature engineering
    periods = [9, 14, 21, 30, 45, 60, 90, 100, 120]

    #Construct technical features for image synthesis
    for period in periods:
        data[f'ema_{period}'] = btalib.ema(data.close, 
                                           period = period).df['ema']
        data[f'ema_{period}_dx'] = np.append(np.nan,np.diff(btalib.ema(data.close, 
                                                                       period = period).df['ema']))
        data[f'rsi_{period}'] = btalib.rsi(data.close, 
                                           period = period).df['rsi']
        data[f'cci_{period}'] = btalib.cci(data.high, 
                                           data.low,  
                                           data.close, 
                                           period = period).df['cci']
        data[f'macd_{period}'] = btalib.macd(data.close, 
                                             pfast = period,
                                             pslow = period*2,
                                             psignal = int(period/3)).df['macd']
        data[f'signal_{period}'] = btalib.macd(data.close, 
                                               pfast = period, 
                                               pslow = period*2,
                                               psignal = int(period/3)).df['signal']
        data[f'hist_{period}'] = btalib.macd(data.close,
                                             pfast = period,
                                             pslow = period*2,
                                             psignal = int(period/3)).df['histogram']
        data[f'volume_{period}'] = btalib.sma(data.volume, 
                                              period = period).df['sma']
        data[f'change_{period}'] = data.close.pct_change(periods = period)

    data = data.drop(data.query('labels == 0').sample(frac=.90).index)

    #Normalize values to 0-1 range
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    #Subset out the labels
    data_trimmed = data.loc[:,  'ema_9':]
    data_trimmed = pd.concat([data_trimmed, data_trimmed.shift(1), data_trimmed.shift(2)], axis = 1)
    # data_trimmed = pd.concat([data_trimmed, data_trimmed.rolling(6).mean(), data_trimmed.rolling(24).mean()], axis = 1)

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    transformed_data = mm_scaler.fit_transform(data_trimmed[24:])
    transformed_data = np.c_[transformed_data, pd.to_numeric(data.labels[24:], downcast = 'signed').to_list()]

    pd.DataFrame(transformed_data).to_csv('ml.csv', index = False)

    return transformed_data