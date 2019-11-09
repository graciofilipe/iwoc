import os
from collections import defaultdict

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
np.random.seed(6)

##For the leads that received one or more calls, how many calls were received on average? [2]
def calls_per_called_lead_fun(calls):
    number_of_unique_called_numbers = calls['Phone Number'].nunique()
    total_number_of_calls = calls.shape[0]
    print('average calls per phone number:', total_number_of_calls / number_of_unique_called_numbers)

# For the leads that signed up, how many calls were received, on average?
def average_calls_per_signed_up_lead_fun(leads, calls, signups):

    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls.loc[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    number_of_unique_called_numbers = calls_to_signed_up_numbers_df['Phone Number'].nunique()
    total_number_of_calls = calls_to_signed_up_numbers_df.shape[0]
    print('average calls per phone number to signed up numbers:',
          total_number_of_calls / number_of_unique_called_numbers)
    print('\n')


# ##Which agent had the most signups? Which assumptions did you make?
# ##Which agent had the most signups per call?
## statiscal significant
def sign_ups_per_agent_fun(leads, calls, signups):
    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads.loc[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls.loc[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

    calls_per_agent_per_sgined_number_df = calls_to_signed_up_numbers_df.groupby(
        by=['Phone Number', 'Agent']).count().drop(
        axis=1, labels=['Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    total_calls_to_number_df = calls_to_signed_up_numbers_df.groupby(by='Phone Number').count().drop(axis=1, labels=[
        'Call Outcome', 'Agent']).rename({'Call Number': 'call_count_per_number'}, axis=1)
    agent_sign_up_counts = defaultdict(int)
    for number in signed_up_phone_numbers_ls:
        total_calls_to_number = total_calls_to_number_df.loc[number][0]
        calls_to_number_by_agents = calls_per_agent_per_sgined_number_df.loc[number]
        for agent in calls_to_number_by_agents.index:
            agent_sign_up_counts[agent] += calls_to_number_by_agents.loc[agent][
                                               0] / total_calls_to_number  # the contribution of the agents is the faction of calls they placed to the signed lead
    print('how many sign up counts per agent:', agent_sign_up_counts)
    print('\n')

    # signs ups per call
    number_of_calls_per_agent_df = calls.groupby(by=['Agent']).count().drop(
        axis=1, labels=['Phone Number', 'Call Outcome']).rename({'Call Number': 'call_count_per_agent'}, axis=1)
    success_rate_per_agent = defaultdict(int)
    for agent in number_of_calls_per_agent_df.index:
        success_rate_per_agent[agent] = agent_sign_up_counts[agent] / number_of_calls_per_agent_df.loc[agent][0]
    print('the success rate per agent:', success_rate_per_agent)
    print('\n')

    # the statistical significance of different agent sign up rates
    number_of_calls_per_agent_ls = [number_of_calls_per_agent_df.loc[agent][0] for agent in
                                    number_of_calls_per_agent_df.index]
    agent_sign_up_counts_ls = [int(agent_sign_up_counts[agent]) for agent in number_of_calls_per_agent_df.index]
    number_of_non_signup_calls_per_agent_ls = [number_of_calls_per_agent_ls[i] - agent_sign_up_counts_ls[i]
                                               for i in range(5)]
    table = [number_of_non_signup_calls_per_agent_ls, agent_sign_up_counts_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    print('pval for the difference between success rates happening by chance', pval)
    print('\n')


## A lead from which region is most likely to be “interested” in the product? [3]
def region_interest_ratio_fun(leads, calls):
    # I'll interpret this as meaning as opposed to "not intested" although one could interpret it differently
    regions = leads['Region'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df = calls_and_leads_df.groupby(by=['Region', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number', 'Agent', 'Call Number', 'Name', 'Sector']).rename({'Age': 'call_count'}, axis=1)
    region_interest_ratio = {
        region: number_of_calls_per_region_and_outcome_df.loc[(region, 'INTERESTED')][0] /
                number_of_calls_per_region_and_outcome_df.loc[(region, 'NOT INTERESTED')][0]
        for region in regions}
    print('region interest ratio', region_interest_ratio)
    print('\n')


##A lead from which sector is most likely to be “interested” in the product? [1]
def sector_interest_ratio_fun(leads, calls):
    sectors = leads['Sector'].unique()
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    number_of_calls_per_region_and_outcome_df = calls_and_leads_df.groupby(by=['Sector', 'Call Outcome']).count().drop(
        axis=1, labels=['Phone Number', 'Agent', 'Call Number', 'Name', 'Region']).rename({'Age': 'call_count'}, axis=1)
    sector_interest_ratio = {
        sector: number_of_calls_per_region_and_outcome_df.loc[(sector, 'INTERESTED')][0] /
                number_of_calls_per_region_and_outcome_df.loc[(sector, 'NOT INTERESTED')][0]
        for sector in sectors}
    print('sector interest ratio', sector_interest_ratio)
    print('\n')


##Given a lead has already expressed interest and signed up:
###signups from which region are most likely to be approved? [2]
###Is this statistically significant? Why? [5]
def region_aproval_ratio_fun(leads, signups):
    signed_up_leads = signups.merge(leads, right_on='Name', left_on='Lead', how='left')
    signed_up_leads_by_region_and_aproval_df = signed_up_leads.groupby(by=['Region', 'Approval Decision']).count()
    regions = leads['Region'].unique()
    approved_counts_per_region_ls = [
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'APPROVED')][0] for region in regions]

    rejected_counts_per_region_ls = [
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'REJECTED')][0] for region in regions]

    region_approval_rate = {regions[i]: approved_counts_per_region_ls[i] / \
                                        (approved_counts_per_region_ls[i] + rejected_counts_per_region_ls[i])
                            for i in range(len(regions))}
    print('region approval rate',region_approval_rate)
    table = [approved_counts_per_region_ls, rejected_counts_per_region_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    print('pval for the difference between region approval rates', pval)
    print('\n')


##Suppose you wanted to pick the 1000 leads most likely to sign up (who have not been called so far), based only on age, sector and region.
###What criteria would you use to pick those leads? [10]
###In what sense are those an optimal criteria set? [3]
###How many signups would you expect to get based on those called leads, assuming they were being called by random agents? [3]
###If you could choose the agents to make those calls, who would you choose? Why? [3]

def most_likely_leads_fun(leads, calls, signups):
    ### DATA PREP
    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    called_numbers_ls = list(calls['Phone Number'])

    leads_and_signups_df = calls_and_leads_df.merge(signups, right_on='Lead', left_on='Name', how='left')
    leads_and_signups_df.drop_duplicates(subset=['Phone Number', 'Name', 'Approval Decision'], inplace=True)
    tmp_modelling_df = leads_and_signups_df.loc[:,['Region', 'Sector', 'Age', 'Approval Decision']]
    tmp_modelling_df.loc[:,'Approval Decision'] = tmp_modelling_df['Approval Decision'].fillna(0)
    tmp_modelling_df.replace(to_replace='APPROVED', value=1, inplace=True)
    tmp_modelling_df.replace(to_replace='REJECTED', value=1, inplace=True)
    tmp_modelling_df = tmp_modelling_df.loc[tmp_modelling_df['Age'] < 90]

    X = tmp_modelling_df.loc[:,['Region', 'Sector', 'Age']]
    region_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()
    X.loc[:,'Region'] = region_encoder.fit_transform(X['Region'])
    X.loc[:,'Sector'] = sector_encoder.fit_transform(X['Sector'])

    y = tmp_modelling_df['Approval Decision']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)

    n_pos = sum(y_train)
    n_neg = sum(y_train == 0)
    print('class balances:  positive:', n_pos/len(y_train), '   negative:', n_neg/len(y_train))
    sample_weights = [1/n_neg if outcome == 0 else 1/n_pos for outcome in y_train]

    ### MODELING FITTING
    clf = RandomForestClassifier(n_estimators=6)
    clf.fit(X=X_train, y=y_train, sample_weight=sample_weights)

    print('results on TRAINING data')
    y_pred_bin = clf.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_bin).ravel()
    print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    print('accuracy', accuracy_score(y_pred=y_pred_bin, y_true=y_train))

    print('results on TEST data')
    y_pred_bin = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    y_test_prob = [test_probs[i][1] for i in range(len(test_probs))]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    print('accuracy', accuracy_score(y_pred=y_pred_bin, y_true=y_test))
    print('auc', roc_auc_score(y_true=y_test, y_score=y_test_prob))


    # ## REUSE ON NON CALLED LEADS ##
    uncalled_leads = leads.loc[~leads['Phone Number'].isin(called_numbers_ls)]
    uncalled_leads.loc[:,'Region'] = region_encoder.fit_transform(uncalled_leads['Region'])
    uncalled_leads.loc[:,'Sector'] = sector_encoder.fit_transform(uncalled_leads['Sector'])
    uncalled_leads.drop(['Name', 'Phone Number'], axis=1, inplace=True)


    probs = clf.predict_proba(uncalled_leads)
    y_prob_for_uncalled_leads = [probs[i][1] for i in range(len(probs))]
    y_prob_for_uncalled_leads.sort()
    prob_of_1000th_lead = y_prob_for_uncalled_leads[-1000]
    print('probability at the 1000th most likely lead:', prob_of_1000th_lead)
    y_test_pred_bin_at_new_threshold = [1 if p > prob_of_1000th_lead else 0 for p in y_test_prob]
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_bin_at_new_threshold).ravel()
    print(
        'the metrics of success/failure for the calls with a probability equal or grater than the top 1000 most likely leads are')
    print('fp', fp, 'tp', tp)
    print('fraction of positives in the 1000 top best leads to call:', tp/(tp+fp))



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

    average_calls_per_signed_up_lead_fun(leads, calls, signups)
    sign_ups_per_agent_fun(leads, calls, signups)
    region_interest_ratio_fun(leads, calls)
    sector_interest_ratio_fun(leads, calls)
    region_aproval_ratio_fun(leads, signups)
    most_likely_leads_fun(leads, calls, signups)


if __name__ == '__main__':
    further_explore()
