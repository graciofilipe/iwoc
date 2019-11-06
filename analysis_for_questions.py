import os
from collections import defaultdict

import pandas as pd
import scipy as sp
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import chi2

def further_exploration(calls, leads, signups, folder=None):

    number_of_unique_called_numbers = calls['Phone Number'].nunique()
    total_number_of_calls = calls.shape[0]
    print('average calls per phone number:', total_number_of_calls / number_of_unique_called_numbers)

    # For the leads that signed up, how many calls were received, on average?
    print('\n')

    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    number_of_unique_called_numbers = calls_to_signed_up_numbers_df['Phone Number'].nunique()
    total_number_of_calls = calls_to_signed_up_numbers_df.shape[0]
    print('average calls per phone number to signed up numbers:',
          total_number_of_calls / number_of_unique_called_numbers)

    # ##Which agent had the most signups? Which assumptions did you make?
    print('\n')

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
    print('\n')

    number_of_calls_per_agent_df = calls.groupby(by=['Agent']).count().drop(
        axis=1, labels=['Phone Number','Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    success_rate_per_agent = defaultdict(int)
    for agent in number_of_calls_per_agent_df.index:
        success_rate_per_agent[agent] = agent_sign_up_counts[agent]/number_of_calls_per_agent_df.loc[agent][0]
    print('the success rate per agent', success_rate_per_agent)

    ## statiscal significant
    print('\n')
    number_of_calls_per_agent_ls = [number_of_calls_per_agent_df.loc[agent][0] for agent in number_of_calls_per_agent_df.index]
    agent_sign_up_counts_ls = [int(agent_sign_up_counts[agent]) for agent in number_of_calls_per_agent_df.index]
    number_of_non_signup_calls_per_agent_ls = [number_of_calls_per_agent_ls[i]-agent_sign_up_counts_ls[i]
                                               for i in range(5)]
    table = [number_of_non_signup_calls_per_agent_ls, agent_sign_up_counts_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    print('pval for the difference between success rates happening by chance', pval)

    ## A lead from which region is most likely to be “interested” in the product? [3]
    print('\n')
    # I'll interpret this as meaning as opposed to "not intested" although one could interpret it differently
    regions = leads['Region'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df  = calls_and_leads_df.groupby(by=['Region', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number',  'Agent',  'Call Number',  'Name',  'Sector']).rename({'Age': 'call_count'}, axis=1)
    region_interest_ratio = {
        region: number_of_calls_per_region_and_outcome_df.loc[(region, 'INTERESTED')][0]/number_of_calls_per_region_and_outcome_df.loc[(region, 'NOT INTERESTED')][0]
                       for region in regions}
    print('region interest ratio', region_interest_ratio)

    print('\n')
    ##A lead from which sector is most likely to be “interested” in the product? [1]
    sectors = leads['Sector'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df  = calls_and_leads_df.groupby(by=['Sector', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number',  'Agent',  'Call Number',  'Name',  'Region']).rename({'Age': 'call_count'}, axis=1)
    sector_interest_ratio = {
        sector: number_of_calls_per_region_and_outcome_df.loc[(sector, 'INTERESTED')][0]/number_of_calls_per_region_and_outcome_df.loc[(sector, 'NOT INTERESTED')][0]
                       for sector in sectors}
    print('sector interest ratio', sector_interest_ratio)

    ##Given a lead has already expressed interest and signed up:
    ###signups from which region are most likely to be approved? [2]
    signed_up_leads = signups.merge(leads, right_on='Name', left_on='Lead', how='left')
    signed_up_leads_by_region_and_aproval_df = signed_up_leads.groupby(by=['Region', 'Approval Decision']).count()

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
