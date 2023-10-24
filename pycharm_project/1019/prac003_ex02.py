import pandas as pd


def bayesian_table(table, prior, likelihood):
    if 'posterior' in table.columns:
        table['prior'] = table['posterior']
    else:
        table['prior'] = prior
    table['likelihood'] = likelihood
    table['joint'] = table['prior'] * table['likelihood']
    table_norm_const = table['joint'].sum()
    table['posterior'] = table['joint'] / table_norm_const

    return table


table = pd.read_csv('PlayTennis.csv')
table.columns = table.columns.str.replace(' ', '_')

columns = list(table.columns)
print(columns)
