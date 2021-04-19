from __future__ import (absolute_import, division, print_function)

import os
import fnmatch
import pandas as pd
import json



class Pipeline:

    @staticmethod
    def parse_and_upload(credfilepath, df, start=False):
        """
        Parses data from google drive via the sheets and drive API's and uploads it to an
        SQL database
        
        Args:
            credfilepath (json): credentials
            sheetname (str): name of sheets file on google drive
            start (boolean): arg to determine if needed
            
        Returns:
            Uploads form response data to a database from google sheets
        """
        if start:
            
            #Send to database
            #Open the connection to the database
            conn = Carrier.pgconnect(credfilepath)

            #Define the SQL statement to insert values into table
            insert_stmt = """INSERT INTO classifier VALUES ( %(id)s, %(name)s, %(suppliers)s, %(postcode)s )"""

            #Ensure a fresh upload of the table
            Carrier.pgquery(conn, "DROP TABLE IF EXISTS classifier CASCADE", msg="cleared old table")
            groupbuy_schema = f"""CREATE TABLE classifier(
                                        {var_string.join(' NUMERIC')} PRIMARY KEY,
                                        name VARCHAR(50),
                                        suppliers VARCHAR(1000),
                                        postcode NUMERIC
                                        );"""
            Carrier.pgquery(conn, groupbuy_schema, msg="created groupbuy table")

            #Inject values from dataframe into database
            Carrier.sqlInject(df, insert_stmt, conn)
            conn.close

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
