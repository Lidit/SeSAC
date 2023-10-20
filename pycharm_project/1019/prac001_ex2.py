import pandas as pd


def bayesian_table(table, prior, likelihood):
    if 'posterior' in table.columns:
        table['prior'] = table['posterior']
    else:
        table['prior'] = prior
    table['likelihood'] = likelihood
    table['unnorm'] = table['prior'] * table['likelihood']
    table_norm_const = table['unnorm'].sum()
    table['posterior'] = table['unnorm'] / table_norm_const

    return table


table = pd.DataFrame(index=['Spam', 'Ham'])

prior = 0.5
likelihood = [0.6, 0.2]
table = bayesian_table(table, prior, likelihood)
print(table)

likelihood = [0.4, 0.05]
table = bayesian_table(table, prior, likelihood)
print(table)
