
seconds = 5000


if seconds > 60:
    minutes = seconds // 60
    if minutes > 60:
        hours = minutes // 60
        minutes -= hours*60
    else:
        hours = 0
    seconds -= (hours*3600 + minutes*60)
else:
    hours = 0
    minutes = 0

print(hours, 'hour', minutes, 'min', seconds, 'sec')
