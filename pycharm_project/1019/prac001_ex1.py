# import pandas as pd
#
# table = pd.DataFrame(index=['Spam', 'Ham'])
#
# table['prior'] = 0.5
# table['likelihood'] = 0.4, 0.05
# table['joint'] = table['prior'] * table['likelihood']
#
# norm_const = table['joint'].sum()
#
# table['posterior'] = table['joint']/norm_const
#
# print(table)

import pandas as pd


def update_bayesian_table(table, likelihood):

    target_table = pd.DataFrame(index=['Spam', 'Ham'])
    target_table['prior'] = table['posterior']
    target_table['likelihood'] = likelihood
    target_table['unnorm'] = target_table['prior'] * target_table['likelihood']
    target_norm_const = target_table['unnorm'].sum()
    target_table['posterior'] = target_table['unnorm'] / target_norm_const
    print(target_table)

    return target_table


link_table = pd.DataFrame(index=['Spam', 'Ham'])

link_table['prior'] = 0.5
link_table['likelihood'] = 0.6, 0.2
link_table['unnorm'] = link_table['prior'] * link_table['likelihood']

link_norm_const = link_table['unnorm'].sum()

link_table['posterior'] = link_table['unnorm'] / link_norm_const

print('Spam/Ham prob with Link')
print(link_table)
print('\n')

link_word_table = pd.DataFrame(index=['Spam', 'Ham'])
link_word_table['prior'] = link_table['posterior']
link_word_table['likelihood'] = 0.4, 0.05
link_word_table['unnorm'] = link_word_table['prior'] * link_word_table['likelihood']

link_word_norm_const = link_word_table['unnorm'].sum()

link_word_table['posterior'] = link_word_table['unnorm'] / link_word_norm_const

print('Spam/Ham prob with Link & word')
print(link_word_table)

update_bayesian_table(link_table, [0.4, 0.05])

# test_table = pd.DataFrame(index=['Spam', 'Ham'])
# print(len(link_table))
# print(test_table.empty)