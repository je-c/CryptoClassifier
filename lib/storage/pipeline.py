from __future__ import (absolute_import, division, print_function)

import os
import fnmatch
import pandas as pd
import json

class Pipeline:

    @staticmethod
    def file_found(date,alt=False, printout=False):
        """ 
        Checks if current date data is already available to the model
            * :param date(DateTime): Date to look for
            * :param alt(bool): Directs search for feature data, (default raw data)
            * :param printout(bool): Text output for delpoyment.

        :return file_found(bool): True if file is found 
        """
        listOfFiles = os.listdir('./data')
        file_found = False

        if not alt:     
            pattern = f"*{date}.csv"    
            for entry in listOfFiles:
                if fnmatch.fnmatch(entry, pattern):
                    file_found = True
            if printout:
                if file_found:
                    print('.csv file found in current directory...')
                    print('Skipping download')
                else:
                    print('File not found...')
                    print('Commencing download')

            return file_found
        else:
            pattern = f'net{date}.csv'
            for entry in listOfFiles:
                if fnmatch.fnmatch(entry, pattern):
                    file_found = True
            if printout:
                if file_found:
                    print(f'net{date}.csv file found in current directory...')
                    print('Skipping generation of network data')
                else:
                    print('File not found...')
                    print('Generating network data')
            return file_found

    @staticmethod
    def push_to_folder(targetDir, dirName, pixels=None, imSize=9, destination='current'):
        """
        Utilised during deployment. Generates and pushes image to test-set directory, also handles push
        to dump folder after prediction.
            * :param targetDir(str): Parent location of target dump folder
            * :param dirName(str): Desired dump folder name
            * :param pixels(np.ndarray): Array of pixel values
            * :param imSize(int): Dimensionality of image (imSize x imSize)
            * :param destination(str): Destination folder for generating images, default current and implementation dependant

        :return currentPath(str): Path to current image directory
        """
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