import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(6)


##For the leads that received one or more calls, how many calls were received on average? [2]
def calls_per_called_lead_fun(calls, file_handle):
    number_of_unique_called_numbers = calls['Phone Number'].nunique()
    total_number_of_calls = calls.shape[0]
    file_handle.write(
        f'average calls per phone number: {np.round(total_number_of_calls / number_of_unique_called_numbers, 2)}\n')
    file_handle.write('\n')


# For the leads that signed up, how many calls were received, on average?
def average_calls_per_signed_up_lead_fun(leads, calls, signups, file_handle):
    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls.loc[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    number_of_unique_called_numbers = calls_to_signed_up_numbers_df['Phone Number'].nunique()
    total_number_of_calls = calls_to_signed_up_numbers_df.shape[0]
    file_handle.write(
        f'average calls per phone number to signed up numbers: {np.round(total_number_of_calls / number_of_unique_called_numbers, 2)}\n')
    file_handle.write('\n')


# ##Which agent had the most signups? Which assumptions did you make?
# ##Which agent had the most signups per call?
## statiscal significant
def sign_ups_per_agent_fun(leads, calls, signups, file_handle):
    agents = list(calls['Agent'].unique())
    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads.loc[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls.loc[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    calls_per_agent_per_sgined_number_df = calls_to_signed_up_numbers_df.groupby(
        by=['Phone Number', 'Agent']).count().drop(
        axis=1, labels=['Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    total_calls_to_number_df = calls_to_signed_up_numbers_df.groupby(by='Phone Number').count().drop(axis=1, labels=[
        'Call Outcome', 'Agent']).rename({'Call Number': 'call_count_per_number'}, axis=1)

    agent_sign_up_counts = {agent: 0 for agent in agents}
    for number in signed_up_phone_numbers_ls:
        total_calls_to_number = total_calls_to_number_df.loc[number][0]
        calls_to_number_by_agents = calls_per_agent_per_sgined_number_df.loc[number]
        for agent in calls_to_number_by_agents.index:
            agent_sign_up_counts[agent] += calls_to_number_by_agents.loc[agent][
                                               0] / total_calls_to_number  # the contribution of the agents is the faction of calls they placed to the signed lead
    agent_sign_up_counts = {k: np.round(v, 1) for k, v in agent_sign_up_counts.items()}
    file_handle.write(f'how many sign up counts per agent: {agent_sign_up_counts}\n')

    # signs ups per call
    number_of_calls_per_agent_df = calls.groupby(by=['Agent']).count().drop(
        axis=1, labels=['Phone Number', 'Call Outcome']).rename({'Call Number': 'call_count_per_agent'}, axis=1)
    success_rate_per_agent = {}
    for agent in number_of_calls_per_agent_df.index:
        success_rate_per_agent[agent] = np.round(
            agent_sign_up_counts[agent] / number_of_calls_per_agent_df.loc[agent][0], 2)
    file_handle.write(f'the success rate per agent:  {success_rate_per_agent}\n')

    # the statistical significance of different agent sign up rates
    number_of_calls_per_agent_ls = [number_of_calls_per_agent_df.loc[agent][0] for agent in
                                    number_of_calls_per_agent_df.index]
    agent_sign_up_counts_ls = [int(agent_sign_up_counts[agent]) for agent in number_of_calls_per_agent_df.index]
    number_of_non_signup_calls_per_agent_ls = [number_of_calls_per_agent_ls[i] - agent_sign_up_counts_ls[i]
                                               for i in range(5)]
    table = [number_of_non_signup_calls_per_agent_ls, agent_sign_up_counts_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    file_handle.write(f'pval for the difference between success rates happening by chance  {pval}\n')
    file_handle.write('\n')


## A lead from which region is most likely to be “interested” in the product? [3]
def region_interest_ratio_fun(leads, calls, file_handle):
    # I'll interpret this as meaning as opposed to "not intested" although one could interpret it differently
    regions = leads['Region'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df = calls_and_leads_df.groupby(by=['Region', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number', 'Agent', 'Call Number', 'Name', 'Sector']).rename({'Age': 'call_count'}, axis=1)
    region_interest_ratio = {
        region: np.round(
            number_of_calls_per_region_and_outcome_df.loc[(region, 'INTERESTED')][0] /
            number_of_calls_per_region_and_outcome_df.loc[(region, 'NOT INTERESTED')][0],
            2)
        for region in regions}
    file_handle.write(f'region interest ratio  {region_interest_ratio}\n')
    file_handle.write('\n')


##A lead from which sector is most likely to be “interested” in the product? [1]
def sector_interest_ratio_fun(leads, calls, file_handle):
    sectors = leads['Sector'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df = calls_and_leads_df.groupby(by=['Sector', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number', 'Agent', 'Call Number', 'Name', 'Region']).rename({'Age': 'call_count'}, axis=1)
    sector_interest_ratio = {
        sector: np.round(
            number_of_calls_per_region_and_outcome_df.loc[(sector, 'INTERESTED')][0] /
            number_of_calls_per_region_and_outcome_df.loc[(sector, 'NOT INTERESTED')][0],
            2)
        for sector in sectors}
    file_handle.write(f'sector interest ratio {sector_interest_ratio}\n')
    file_handle.write('\n')


##Given a lead has already expressed interest and signed up:
###signups from which region are most likely to be approved? [2]
###Is this statistically significant? Why? [5]
def region_aproval_ratio_fun(leads, signups, file_handle):
    signed_up_leads = signups.merge(leads, right_on='Name', left_on='Lead', how='left')
    signed_up_leads_by_region_and_aproval_df = signed_up_leads.groupby(by=['Region', 'Approval Decision']).count()
    regions = leads['Region'].unique()
    approved_counts_per_region_ls = [
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'APPROVED')][0] for region in regions]

    rejected_counts_per_region_ls = [
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'REJECTED')][0] for region in regions]

    region_approval_rate = {regions[i]: np.round(
        approved_counts_per_region_ls[i] / (approved_counts_per_region_ls[i] + rejected_counts_per_region_ls[i]),
        2) for i in range(len(regions))}
    file_handle.write(f'region approval rate {region_approval_rate} \n')
    table = [approved_counts_per_region_ls, rejected_counts_per_region_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    file_handle.write(f'pval for the difference between region approval rates {pval} \n')
    file_handle.write('\n')


##Suppose you wanted to pick the 1000 leads most likely to sign up (who have not been called so far), based only on age, sector and region.
###What criteria would you use to pick those leads? [10]
###In what sense are those an optimal criteria set? [3]
###How many signups would you expect to get based on those called leads, assuming they were being called by random agents? [3]
###If you could choose the agents to make those calls, who would you choose? Why? [3]

def most_likely_leads_fun(leads, calls, signups, file_handle):
    ### DATA PREP
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    called_numbers_ls = list(calls['Phone Number'])

    leads_and_signups_df = calls_and_leads_df.merge(signups, right_on='Lead', left_on='Name', how='left')
    leads_and_signups_df.drop_duplicates(subset=['Phone Number', 'Name', 'Approval Decision'], inplace=True)
    tmp_modelling_df = leads_and_signups_df.loc[:, ['Region', 'Sector', 'Age', 'Approval Decision']]
    tmp_modelling_df.loc[:, 'Approval Decision'] = tmp_modelling_df['Approval Decision'].fillna(0)
    tmp_modelling_df.replace(to_replace='APPROVED', value=1, inplace=True)
    tmp_modelling_df.replace(to_replace='REJECTED', value=1, inplace=True)
    tmp_modelling_df = tmp_modelling_df.loc[tmp_modelling_df['Age'] < 90]

    X = tmp_modelling_df.loc[:, ['Region', 'Sector', 'Age']]
    region_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()
    X.loc[:, 'Region'] = region_encoder.fit_transform(X['Region'])
    X.loc[:, 'Sector'] = sector_encoder.fit_transform(X['Sector'])

    y = tmp_modelling_df['Approval Decision']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)

    n_pos = sum(y_train)
    n_neg = sum(y_train == 0)
    file_handle.write(
        f'class balances:  positive:  {np.round(n_pos / len(y_train), 2)}   negative: {np.round(n_neg / len(y_train), 2)}\n')
    sample_weights = [1 / n_neg if outcome == 0 else 1 / n_pos for outcome in y_train]

    ### MODELING FITTING
    clf = RandomForestClassifier(n_estimators=6)
    clf.fit(X=X_train, y=y_train, sample_weight=sample_weights)

    file_handle.write('results on TEST data \n')

    y_pred_bin = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    y_test_prob = [test_probs[i][1] for i in range(len(test_probs))]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    file_handle.write(f'tn:{tn}  fp:{fp}  fn:{fn}  tp:{tp} \n')
    file_handle.write(f'accuracy {np.round(accuracy_score(y_pred=y_pred_bin, y_true=y_test), 2)} \n')
    file_handle.write(f'auc  {np.round(roc_auc_score(y_true=y_test, y_score=y_test_prob), 2)} \n')

    # ## REUSE ON NON CALLED LEADS ##
    uncalled_leads = leads.loc[~leads['Phone Number'].isin(called_numbers_ls)]
    uncalled_leads.loc[:, 'Region'] = region_encoder.fit_transform(uncalled_leads['Region'])
    uncalled_leads.loc[:, 'Sector'] = sector_encoder.fit_transform(uncalled_leads['Sector'])
    uncalled_leads.drop(['Name', 'Phone Number'], axis=1, inplace=True)

    probs = clf.predict_proba(uncalled_leads)
    y_prob_for_uncalled_leads = [probs[i][1] for i in range(len(probs))]
    y_prob_for_uncalled_leads.sort()
    prob_of_1000th_lead = y_prob_for_uncalled_leads[-1000]
    file_handle.write(f'probability at the 1000th most likely lead: {np.round(prob_of_1000th_lead, 2)}\n')
    y_test_pred_bin_at_new_threshold = [1 if p > prob_of_1000th_lead else 0 for p in y_test_prob]
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_bin_at_new_threshold).ravel()
    file_handle.write(
        'the metrics of success/failure for the calls with a probability equal or grater than the top 1000 most likely leads are\n')
    file_handle.write(f'p:{fp}  tp:{tp} \n')
    file_handle.write(f'fraction of positives in the 1000 top best leads to call:  {np.round(tp / (tp + fp), 2)}\n')
    file_handle.write('\n')


def further_explore():
    """
    execute the exploration pipeline
    :return:
    """
    calls = pd.read_csv('input_data/calls.csv')
    leads = pd.read_csv('input_data/leads.csv')
    signups = pd.read_csv('input_data/signups.csv')

    output_folder = 'QA'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_handle = open(f'{output_folder}/analysis_logs.txt', 'w')
    calls_per_called_lead_fun(calls, file_handle)
    average_calls_per_signed_up_lead_fun(leads, calls, signups, file_handle)
    sign_ups_per_agent_fun(leads, calls, signups, file_handle)
    region_interest_ratio_fun(leads, calls, file_handle)
    sector_interest_ratio_fun(leads, calls, file_handle)
    region_aproval_ratio_fun(leads, signups, file_handle)
    most_likely_leads_fun(leads, calls, signups, file_handle)
    file_handle.close()


if __name__ == '__main__':
    further_explore()
