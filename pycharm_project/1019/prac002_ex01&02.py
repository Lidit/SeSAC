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


table = pd.DataFrame(index=['X', 'Y'])

prior = 0.5
likelihood = [0.1, 0.8]
table = bayesian_table(table, prior, likelihood)
print('Probability when took 1 ball and color of ball is B')
print(table['posterior'], '\n')

table = pd.DataFrame(index=['X', 'Y'])

prior = 0.5
likelihood = [0.9, 0.2]
table = bayesian_table(table, prior, likelihood)
print('Probability when took 1 ball and color of ball is W')
print(table['posterior'])
