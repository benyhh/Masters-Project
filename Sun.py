# //
import time
import ephem

## Site Definition #########
apex = ephem.Observer()

apex.lat       = '-23:00:20.8'
apex.long      = '-67:45:33.0'
apex.elev      =   5105

planet = ephem.Jupiter()
sun    = ephem.Sun()


apex.date = time.strftime('2020/09/01 00:00')

DATE0 = apex.date

dt = 1*ephem.hour

N = 365*24*2

for i in range(N):

    T = DATE0+i*dt
    apex.date = T

    sun.compute( apex )

    Az = sun.az  * 180 / 3.14
    El = sun.alt * 180 / 3.14

    print( i, apex.date, Az, El)
