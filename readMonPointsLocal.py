import pymysql as mysql
import pandas  as pd
from tqdm import tqdm

def readMONITOR( db_table , date0 , date1 ):

    keyname = db_table.split( ':' )[ -1 ].rstrip()
    db = 'MONITOR'
    conn  = mysql.connect( host   = 'localhost' ,
                           user   = 'root'  ,
                           passwd = 'bendikny'  ,

                           db     = db    )

    query = f'SELECT * FROM {db}.`{db_table}` WHERE `TS` >= \'{date0}\' \
                                AND   `TS` <  \'{date1}\' \
                                order by TS asc'

    df    = pd.read_sql_query( query , conn )

    df.columns  = [ 'date' , 'value' ]
    df['value'] = df.value.astype('float')
    df.rename( columns = { 'value' : keyname } , inplace = 'True' )

    df['date'] = pd.to_datetime(df['date'])

    conn.close()

    return df


mpfile  = './Data/monpoints'
pdbfile = './Data/PointingTable.csv'

with open( mpfile , 'r' ) as f :
    MONPOINTS = f.readlines()


pointingScans = pd.read_csv(  pdbfile                       \
                            , infer_datetime_format =  True)


print("Here", pointingScans.columns)

pointingScans.rename( columns = {'obs_date':'date'} , inplace = True )
pointingScans[ 'date' ] = pd.to_datetime( pointingScans[ 'date' ] )

dt1 = pd.Timedelta( minutes = 15 )
dt2 = pd.Timedelta( minutes = 3 )

DF2 = pd.DataFrame()
DF3 = pd.DataFrame()

print(len(pointingScans))
for j, date0 in enumerate(tqdm(pointingScans.date)):

    for i, monpoint in enumerate(MONPOINTS) :

        df = readMONITOR( monpoint.rstrip() , date0-dt1 , date0+dt2 )

        if i == 0:
            DF = df
        else :
            DF = pd.merge( DF , df , left_on = [ 'date' ] , right_on = [ 'date' ] , how = 'outer' , sort = True )            

    DF['scan']=pointingScans.scan[j]
    DF2 = DF2.append( DF )


DF2 = DF2.set_index('date')
DF2 = DF2.drop_duplicates()

DF2.to_csv( 'test_query.csv' , header = True )




### SEARCH POINTING DB USING THE FOLLOWING ONE LINER :

### mysql -h apexmysql2 -P 3306 -u SciopsReader -psciopsro -e "use SciopsDB; SELECT obs_date,scan from redPointDB where obs_date > '2022-08-01' ORDER BY obs_date"  | tr '\t' ',' > pointingScans.txt