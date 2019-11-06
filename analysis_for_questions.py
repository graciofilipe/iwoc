import os
from collections import defaultdict

import pandas as pd
import scipy as sp
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import chi2

def further_exploration(calls, leads, signups, folder=None):
    leads_and_calls = calls.merge(leads, how='outer', on=['Phone Number'])

    number_of_unique_called_numbers = calls['Phone Number'].nunique()
    total_number_of_calls = calls.shape[0]
    print('average calls per phone number:', total_number_of_calls / number_of_unique_called_numbers)

    # For the leads that signed up, how many calls were received, on average?
    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    number_of_unique_called_numbers = calls_to_signed_up_numbers_df['Phone Number'].nunique()
    total_number_of_calls = calls_to_signed_up_numbers_df.shape[0]
    print('average calls per phone number to signed up numbers:',
          total_number_of_calls / number_of_unique_called_numbers)

    # ##Which agent had the most signups? Which assumptions did you make?
    calls_per_agent_per_sgined_number_df = calls_to_signed_up_numbers_df.groupby(by=['Phone Number', 'Agent']).count().drop(
        axis=1, labels=['Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    total_calls_to_number_df = calls_to_signed_up_numbers_df.groupby(by='Phone Number').count().drop(axis=1, labels=[
        'Call Outcome', 'Agent']).rename({'Call Number': 'call_count_per_number'}, axis=1)
    agent_sign_up_counts = defaultdict(int)
    for number in signed_up_phone_numbers_ls:
        calls_to_number_by_agent = calls_per_agent_per_sgined_number_df.loc[number]
        total_calls_to_number = total_calls_to_number_df.loc[number][0]
        for agent in calls_to_number_by_agent.index:
            agent_sign_up_counts[agent] += calls_to_number_by_agent.loc[agent][
                                               0] / total_calls_to_number  # the contribution of the agents is the faction of calls they placed to the lead
    print('how many sign up counts per agent:', agent_sign_up_counts)

    ##Which agent had the most signups per call?
    number_of_calls_per_agent_df = calls.groupby(by=['Agent']).count().drop(
        axis=1, labels=['Phone Number','Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    success_rate_per_agent = defaultdict(int)
    for agent in number_of_calls_per_agent_df.index:
        success_rate_per_agent[agent] = agent_sign_up_counts[agent]/number_of_calls_per_agent_df.loc[agent][0]
    print('the success rate per agent', success_rate_per_agent)

    ## statiscal significant
    number_of_calls_per_agent_ls = [number_of_calls_per_agent_df.loc[agent][0] for agent in number_of_calls_per_agent_df.index]
    agent_sign_up_counts_ls = [int(agent_sign_up_counts[agent]) for agent in number_of_calls_per_agent_df.index]
    table = [number_of_calls_per_agent_ls, agent_sign_up_counts_ls]
    stat, p, dof, expected = chi2_contingency(table)
    print('dof=%d' % dof)
    print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')

    import ipdb; ipdb.set_trace()



def further_explore():
    """
    execute the exploration pipeline
    :return:
    """
    calls = pd.read_csv('input_data/calls.csv')
    leads = pd.read_csv('input_data/leads.csv')
    signups = pd.read_csv('input_data/signups.csv')

    output_folder = 'exploratory_analysis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    further_exploration(calls=calls,
                        leads=leads,
                        signups=signups)


if __name__ == '__main__':
    further_explore()
