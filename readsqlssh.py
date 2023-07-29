import pandas as pd
import pymysql
import sys
import logging
import socket
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import time
sys.path.insert(1, 'C:/Users/bendi\PP/ting')
from ting import pwuio

ssh_host = 'login.astro.uio.no'
ssh_username = 'bendikny'
ssh_password = pwuio
database_username = 'apex'
database_password = 'apecs4op'
database_name = 'MONITOR'
localhost = '127.0.0.1'
db     = 'MONITOR'



query = 'SELECT TOP 100 FROM `%s` WHERE `TS` >= \'%s\' \
                            AND   `TS` <  \'%s\' \
                            order by TS asc'     \
                    % ( "ABM1:ANTMOUNT:ACTUALAZ" , "2022-07-01 09:04:43" , "2022-07-01 09:08:43" ) 

def open_ssh_tunnel(verbose=True):
    """Open an SSH tunnel and connect using a username and password.
    
    :param verbose: Set to True to show logging
    :return tunnel: Global SSH tunnel connection
    """
    
    if verbose:
        sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG
    
    global tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host, 22),
        ssh_username = ssh_username,
        ssh_password = ssh_password,
        remote_bind_address=('127.0.0.1', 8080),
        local_bind_address=('127.0.0.1', 8080)
        )
    print("Starting")
    tunnel.start()
    print(tunnel.local_bind_port)
    time.sleep(5)


def mysql_connect():
    """Connect to a MySQL server using the SSH tunnel connection
    
    :return connection: Global MySQL database connection
    """
    
    global connection
    
    connection = pymysql.connect(
        host='127.0.0.1',
        user=database_username,
        passwd=database_password,
        db=database_name,
        port=tunnel.local_bind_port
    )


def run_query(sql):
    """Runs a given SQL query via the global database connection.
    
    :param sql: MySQL query
    :return: Pandas dataframe containing results
    """
    
    return pd.read_sql_query(sql, connection)


def mysql_disconnect():
    """Closes the MySQL database connection.
    """
    
    connection.close()

def close_ssh_tunnel():
    """Closes the SSH tunnel connection.
    """
    
    tunnel.close

open_ssh_tunnel()
mysql_connect()
#df = run_query(query)
#df.head()

#mysql_disconnect()
close_ssh_tunnel()