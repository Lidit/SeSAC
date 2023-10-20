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


table = pd.read_csv('PlayTennis.csv')
table.columns = table.columns.str.replace(' ', '_')

hypothesis_y = table.loc[table.Play_Tennis == 'Yes', :]
n_hypothesis_y = len(hypothesis_y)
likelihood_wind_y = hypothesis_y['Wind'].value_counts() / n_hypothesis_y

print('Hypothesis in Yes, likelihood in Wind')
print(likelihood_wind_y, '\n')

hypothesis_n = table.loc[table.Play_Tennis == 'No', :]
n_hypothesis_n = len(hypothesis_n)
likelihood_wind_n = hypothesis_n['Wind'].value_counts() / n_hypothesis_n

print('Hypothesis in No, likelihood in Wind')
print(likelihood_wind_n, '\n')

likelihood_outlook_y = hypothesis_y['Outlook'].value_counts() / n_hypothesis_y

print('Hypothesis in Yes, likelihood in Outlook')
print(likelihood_outlook_y)


# Get posterior

# prior = list(table['Play_Tennis'].value_counts() / len(table))
# print(table['Play_Tennis'].value_counts())
# print(prior)
#
# wind_table = pd.DataFrame(index=['Yes', 'No'])
# likelihood = likelihood_wind_y[0], likelihood_wind_n[1]
#
# print(likelihood)
#
# wind_table_y = pd.DataFrame(index=['Weak', 'Strong'])
#
# prior = likelihood_wind_y
# wind_table_y = bayesian_table(wind_table_y, prior, likelihood_wind_y)
#
# print(wind_table_y)
#
# wind_table_n = pd.DataFrame(index=['Weak', 'Strong'])
# prior = likelihood_wind_n
# wind_table_n = bayesian_table(wind_table_n, prior, likelihood_wind_n)
#
# print(wind_table_n)
#
