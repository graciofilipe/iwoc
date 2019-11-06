import os
from collections import defaultdict

import pandas as pd


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
    calls_per_agent_per_number_df = calls_to_signed_up_numbers_df.groupby(by=['Phone Number', 'Agent']).count().drop(
        axis=1, labels=['Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    total_calls_to_number_df = calls_to_signed_up_numbers_df.groupby(by='Phone Number').count().drop(axis=1, labels=[
        'Call Outcome', 'Agent']).rename({'Call Number': 'call_count_per_number'}, axis=1)
    agent_sign_up_counts = defaultdict(int)
    for number in signed_up_phone_numbers_ls:
        calls_to_number_by_agent = calls_per_agent_per_number_df.loc[number]
        total_calls_to_number = total_calls_to_number_df.loc[number][0]
        for agent in calls_to_number_by_agent.index:
            agent_sign_up_counts[agent] += calls_to_number_by_agent.loc[agent][
                                               0] / total_calls_to_number  # the contribution of the agents is the faction of calls they placed to the lead
    print('how many sign up counts per agent:', agent_sign_up_counts)


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
