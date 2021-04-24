import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import json
import pandas as pd

class SQLTools:
    """
    SQLTools Class
    --------------

    Wrapper for SQL and databasing tools. Contains functions for querying non-local and local databases,
    flexible schema mapping and data storage.
    """
    
    @staticmethod
    def pgquery(conn, sqlcmd, args=None, msg=False, returntype='tuple'):
        """ 
        Utility function for packaging SQL statements from python.
            * :param conn(psycopg2.connection): Connection port to SQL database
            * :param sqlcmd(str): SQL query
            * :param args(dict): Optional arguements for SQL-side
            * :param msg(str): Return message from server
            * :param returntype(str): Demarkation of expected query return type

        :return returnval(str): Return message
        """
        returnval = None
        with conn:
            cursortype = None if returntype != 'dict' else psycopg2.extras.RealDictCursor
            with conn.cursor(cursor_factory=cursortype) as cur:
                try:
                    if args is None:
                        cur.execute(sqlcmd)
                    else:
                        cur.execute(sqlcmd, args)
                    if (cur.description != None ):
                        returnval = cur.fetchall()
                    if msg != False:
                        print("success: " + msg)
                except psycopg2.DatabaseError as e:
                    if e.pgcode != None:
                        if msg: print("db read error: "+msg)
                        print(e)
                except Exception as e:
                    print(e)
        return returnval

    @staticmethod
    def pgconnect(credential_filepath):
        """ 
        Connection terminal from python to SQL server
            * :param credential_filepath(str): Filepath to credentials.json

        :return conn(psycopg2.connection): Connection port to SQL server
        """
        try:
            with open(credential_filepath) as f:
                db_conn_dict = json.load(f)
            conn = psycopg2.connect(**db_conn_dict)
            print('Connection successful')
        except Exception as e:
            print("Connection unsuccessful... Try again")
            print(e)
            return None
        return conn
    
    @staticmethod
    def sqlInject(data, insert_stmt, conn):
        """ 
        Pipeline for uploading data to SQL server
            * :param data(pd.DataFrame): Data to upload
            * :param insert_stmt(str): SQL query
            * :param conn(psycopg2.connection): Connection port to SQL server

        :return (NoneType): None
        """
        count = 0
        if isinstance(data, list):
            for df in data:
                print(f'Element {count} data injection commenced')
                
                for _, area in df.iterrows():
                    SQLTools.pgquery(conn, insert_stmt, args=area, msg="inserted ")
                
                count += 1
                
        else:
            for _, area in data.iterrows():
                SQLTools.pgquery(conn, insert_stmt, args=area, msg="inserted ")

    @staticmethod
    def feed_schema(data):
        """ 
        Flexible schema mapping for incoming data. Allows non-fixed axis 1 dimensions
            * :param data(pd.DataFrame): Data to upload

        :return (str): Schema segment to splice into table creation schema 
        """
        schema_segment = [
            f'{col} NUMERIC,\n' if col != data.columns()[-1] else f'{col} NUMERIC' for col in data.columns()
        ]
        return ''.join(schema_segment)

    @staticmethod
    def parse_and_upload(credfilepath, data, start=False):
        """
        Parses data from local directories to relational database.
            * :param credfilepath(json): Database credentials
            * :param data(pd.DataFrame): Data to upload
            * :param start(boolean): arg to determine if needed
        
        :return (NoneType): None
        """
        if start:
            conn = SQLTools.pgconnect(credfilepath)

            insert_stmt = f"""
                INSERT INTO classifier VALUES ( {', '.join([col for col in data.columns()])} )
            """
            
            SQLTools.pgquery(conn, "DROP TABLE IF EXISTS classifier CASCADE", msg="cleared old table")
            groupbuy_schema = f"""
                    CREATE TABLE classifier(
                        date DATETIME PRIMARY KEY,
                        {SQLTools.feed_schema(data)}
                    );
            """
            SQLTools.pgquery(conn, groupbuy_schema, msg="created groupbuy table")
            SQLTools.sqlInject(data, insert_stmt, conn)

            conn.close