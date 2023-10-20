import pandas as pd

table = pd.DataFrame(index=['Spam', 'Ham'])

table['prior'] = 0.5
table['likelihood'] = 0.6, 0.2
table['joint'] = table['prior'] * table['likelihood']

norm_const = table['joint'].sum()

table['posterior'] = table['joint']/norm_const

print(table)


