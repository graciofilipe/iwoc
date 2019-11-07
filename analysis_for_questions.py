import os
from collections import defaultdict

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import scipy as sp

#
# def further_exploration(calls, leads, signups, folder=None):
#
#     number_of_unique_called_numbers = calls['Phone Number'].nunique()
#     total_number_of_calls = calls.shape[0]
#     print('average calls per phone number:', total_number_of_calls / number_of_unique_called_numbers)

# For the leads that signed up, how many calls were received, on average?
def average_calls_per_signed_up_lead_fun(leads, calls, signups):
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
# ##Which agent had the most signups per call?
## statiscal significant


def sign_ups_per_agent_fun(leads, calls, signups):
    signed_up_names_ls = list(signups['Lead'])
    signed_up_leads_df = leads[leads['Name'].isin(signed_up_names_ls)]
    signed_up_phone_numbers_ls = list(signed_up_leads_df['Phone Number'])
    calls_to_signed_up_numbers_df = calls[calls['Phone Number'].isin(signed_up_phone_numbers_ls)]

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

    print('\n')

    number_of_calls_per_agent_df = calls.groupby(by=['Agent']).count().drop(
        axis=1, labels=['Phone Number','Call Outcome']).rename({'Call Number': 'call_count_per_agent_per_number'}, axis=1)
    success_rate_per_agent = defaultdict(int)
    for agent in number_of_calls_per_agent_df.index:
        success_rate_per_agent[agent] = agent_sign_up_counts[agent]/number_of_calls_per_agent_df.loc[agent][0]
    print('the success rate per agent', success_rate_per_agent)

    print('\n')
    number_of_calls_per_agent_ls = [number_of_calls_per_agent_df.loc[agent][0] for agent in number_of_calls_per_agent_df.index]
    agent_sign_up_counts_ls = [int(agent_sign_up_counts[agent]) for agent in number_of_calls_per_agent_df.index]
    number_of_non_signup_calls_per_agent_ls = [number_of_calls_per_agent_ls[i]-agent_sign_up_counts_ls[i]
                                               for i in range(5)]
    table = [number_of_non_signup_calls_per_agent_ls, agent_sign_up_counts_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    print('pval for the difference between success rates happening by chance', pval)


## A lead from which region is most likely to be “interested” in the product? [3]
def region_interest_ratio_fun(leads, calls):
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


##A lead from which sector is most likely to be “interested” in the product? [1]
def sector_interest_ratio_fun(leads, calls):
    print('\n')
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
###Is this statistically significant? Why? [5]
def region_aproval_ratio_fun(leads, signups):
    signed_up_leads = signups.merge(leads, right_on='Name', left_on='Lead', how='left')
    signed_up_leads_by_region_and_aproval_df = signed_up_leads.groupby(by=['Region', 'Approval Decision']).count()
    regions = leads['Region'].unique()
    approved_counts_per_region_ls =[
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'APPROVED')][0] for region in regions]

    rejected_counts_per_region_ls =[
        signed_up_leads_by_region_and_aproval_df.loc[(region, 'REJECTED')][0] for region in regions]\

    region_approval_rate = {regions[i]: approved_counts_per_region_ls[i]/ \
                                        (approved_counts_per_region_ls[i]+rejected_counts_per_region_ls[i])
                            for i in range(len(regions))}
    print(region_approval_rate)
    table = [approved_counts_per_region_ls, rejected_counts_per_region_ls]
    stat, pval, dof, expected = chi2_contingency(table)
    print('pval for the difference between region approval rates', pval)


##Suppose you wanted to pick the 1000 leads most likely to sign up (who have not been called so far), based only on age, sector and region.
###What criteria would you use to pick those leads? [10]
###In what sense are those an optimal criteria set? [3]
###How many signups would you expect to get based on those called leads, assuming they were being called by random agents? [3]
###If you could choose the agents to make those calls, who would you choose? Why? [3]

def most_likely_leads_fun(leads, calls, signups):

    # redo this with leads that have been called only

    calls_and_leads_df = calls.merge(leads, on='Phone Number', how='left')
    leads_and_signups_df = calls_and_leads_df.merge(signups, right_on='Lead', left_on='Name', how='left')
    leads_and_signups_df.drop_duplicates(subset=['Phone Number', 'Name', 'Approval Decision'], inplace=True)
    tmp_modelling_df = leads_and_signups_df[['Region', 'Sector',  'Age', 'Approval Decision']]
    tmp_modelling_df['Approval Decision'] = tmp_modelling_df['Approval Decision'].fillna(0)
    tmp_modelling_df.replace(to_replace='APPROVED', value=1, inplace=True)
    tmp_modelling_df.replace(to_replace='REJECTED', value=1, inplace=True)

    region_encoder = OneHotEncoder()
    region_encoded = region_encoder.fit_transform(tmp_modelling_df.Region.values.reshape(-1, 1)).toarray()
    region_df = pd.DataFrame(data=region_encoded,
                             columns=[ca for ca in region_encoder.categories_[0]],
                             index=tmp_modelling_df.index.values)
    sector_encoder = OneHotEncoder()
    sector_encoded = sector_encoder.fit_transform(tmp_modelling_df.Sector.values.reshape(-1, 1)).toarray()
    sector_df = pd.DataFrame(data=sector_encoded,
                             columns=[ca for ca in sector_encoder.categories_[0]],
                             index=tmp_modelling_df.index.values)

    X = pd.concat([sector_df, region_df], axis=1)
    X['age'] = tmp_modelling_df['Age']
    y = tmp_modelling_df['Approval Decision']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)
    logistic_reg = linear_model.LogisticRegressionCV(fit_intercept=False)
    n_pos = sum(y_train)
    n_neg = sum(y_train==0)
    sample_weights = [1/n_neg if outcome ==0 else 1/n_pos for outcome in y_train]
    logistic_reg.fit(X=X_train, y=y_train, sample_weight=sample_weights)

    print('results on TRAINING data')

    y_pred_bin = logistic_reg.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_bin).ravel()
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)
    print('accuracy', accuracy_score(y_pred=y_pred_bin, y_true=y_train))

    print('results on TEST data')
    y_pred_bin = logistic_reg.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)
    print('accuracy', accuracy_score(y_pred=y_pred_bin, y_true=y_test))


    import ipdb; ipdb.set_trace()




    # dont forget to assert column order is the same
    return 1



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

    # average_calls_per_signed_up_lead_fun(leads, calls, signups)
    # sign_ups_per_agent_fun(leads, calls, signups)
    # region_interest_ratio_fun(leads, calls)
    # sector_interest_ratio_fun(leads, calls)
    # region_aproval_ratio_fun(leads, signups)
    most_likely_leads_fun(leads, calls, signups)

if __name__ == '__main__':
    further_explore()
