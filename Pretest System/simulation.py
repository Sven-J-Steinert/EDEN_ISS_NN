from random import randrange
import os
import datetime
import time

# Status (0,1)  Heater [H], Window [W], Fan [F]
# Value e R     Temp [T], Velocity [V], CO2 [C]
# datavector(time[t]) [H,W,F,T,V,C]

# startparameter
t = 0
H = 0
W = 0
F = 1
T = 20
V = 3
C = 1000

data_v = [H,W,F,T,V,C]

seconds = 0
minutes = 0
hours = 0

datestring = datetime.datetime.now().strftime("%Y-%m-%d ")
timestamp = datestring + str(hours) + ':' + str(minutes) + ':' + str(seconds)


Fan_counter = 100

Window_counter = 0

print('setup')
# delete old .csv
os.remove("pretest.csv")
# create new .csv
with open('pretest.csv','a') as writer:
    # writer.write('t;H;W;F;T;V;C\n')
    writer.write(str(timestamp) + ';' + str(H) + ';' + str(W) + ';' + str(F) + ';' + str(T) + ';' + str(V) + ';' + str(C) + '\n')

print(str(t) + ' ' + str(data_v))

print('starting simulation...')

# 1 day = 86400
for t in range(1,10000):

    # regulation rules
    if T < 20:
        H = 1
    else:
        H = 0

    if ( C > 1050 or T > 25 ) and W == 0 :
        W = 1
        Window_counter = 60
    elif Window_counter == 0:
        W = 0

    if ( minutes == 0 or minutes == 30 ) and seconds == 0 and F == 0:
        F = 1
        Fan_counter = 100
    elif Fan_counter == 0:
        F = 0

    # INFLUENCES
    # external human influence
    C = C + 0.5 + 0.3*(randrange(10)-5)
    if T < 36:
        T = T + 0.001 + 0.001*(randrange(10)-5)

    # Fan influence
    if F == 0 and W == 0:
        V = 0 + 0.01*randrange(10)
        exchange_rate = 1 + V/3
    if F == 1 and W == 0:
        V = 3 + 0.01*(randrange(10)-5)
        exchange_rate = 1 + V/3
    if F == 1 and W == 1:
        V = 4 + 0.01*(randrange(10)-5)
        exchange_rate = 1 + V/3

    # Window influence
    if W == 1:
        if F == 0:
            V = 1 + 0.01*(randrange(10)-5)
            exchange_rate = 1 + V/3
        if T > 10:
            T = T - ( 0.1 + 0.02*(randrange(10)-5))*exchange_rate
        if C > 400:
            C = C - ( 4 + 0.5*(randrange(10)-5))*exchange_rate

    # Heater influence
    if H == 1:
        T = T + (0.05 + 0.01*(randrange(10)-5))*exchange_rate



    # clean data
    T = round(T,4)
    V = round(V,2)
    C = round(C,2)
    data_v = [H,W,F,T,V,C]
    #print(str(t) + ' ' + str(data_v))

    seconds = seconds + 1
    if seconds >= 60:
        minutes = minutes + 1
        seconds = 0
    if minutes >= 60:
        hours = hours + 1
        minutes = 0

    timestamp = datestring + str(hours) + ':'  + str(minutes) + ':' + str(seconds)
    # save new line in .csv
    with open('pretest.csv','a') as writer:
        writer.write(str(timestamp) + ';' + str(H) + ';' + str(W) + ';' + str(F) + ';' + str(T) + ';' + str(V) + ';' + str(C) + '\n')

    # 1 second timestep
    t = t+1

    if Window_counter > 0:
        Window_counter = Window_counter -1

    if Fan_counter > 0:
        Fan_counter = Fan_counter -1


print('end')
