from random import randrange
import os
import datetime
import time

timestamp = ''
step_size = 5  # in min

# start date
years = 2018
months = 2
days = 7

hours = 0
minutes = 0
seconds = 0

# end date
end_years = 2018
end_months = 8
end_days = 13

end_hours = 14
end_minutes = 0
end_seconds = 0


os.remove('timeframe.csv')

timestamp = str(years) + '-' + str(f"{months:02d}") + '-' + str(f"{days:02d}") + ' ' + str(f"{hours:02d}") + ':' + str(f"{minutes:02d}") + ':' + str(f"{seconds:02d}")

print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('start date', end=" ")
print(timestamp)
print('  end date', end=" ")
print(str(end_years) + '-' + str(f"{end_months:02d}") + '-' + str(f"{end_days:02d}") + ' ' + str(f"{end_hours:02d}") + ':' + str(f"{end_minutes:02d}") + ':' + str(f"{end_seconds:02d}"), end=' ')
print('     step size: ' + str(step_size) + 'min')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('WRITING timeframe.csv', end=" ", flush = True)

with open('timeframe.csv','a') as writer:
    writer.write('Date_Time' + ';' + 'Placeholder' + '\n')
    writer.write(str(timestamp) + ';' + str(0) + '\n')

while (years != end_years) or (months != end_months) or (days != end_days) or (hours != end_hours) or (minutes != end_minutes) or (seconds != end_seconds):

    minutes = minutes + step_size

    if seconds >= 60:
        minutes = minutes + 1
        seconds = 0
    if minutes >= 60:
        hours = hours + 1
        minutes = 0
    if hours >= 24:
        days = days + 1
        hours = 0


    # calendar 2018
    if (months == 1) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 2) and (days >= 28):
        months = months + 1
        days = 1
    if (months == 3) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 4) and (days >= 30):
        months = months + 1
        days = 1
    if (months == 5) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 6) and (days >= 30):
        months = months + 1
        days = 1
    if (months == 7) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 8) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 9) and (days >= 30):
        months = months + 1
        days = 1
    if (months == 10) and (days >= 31):
        months = months + 1
        days = 1
    if (months == 11) and (days >= 30):
        months = months + 1
        days = 1
    if (months == 12) and (days >= 31):
        years = years + 1
        months = 1
        days = 1

    timestamp = str(years) + '-' + str(f"{months:02d}") + '-' + str(f"{days:02d}") + ' ' + str(f"{hours:02d}") + ':' + str(f"{minutes:02d}") + ':' + str(f"{seconds:02d}")

    with open('timeframe.csv','a') as writer:
        writer.write(str(timestamp) + ';' + str(0) + '\n')

print('     done.')

print('end.')
