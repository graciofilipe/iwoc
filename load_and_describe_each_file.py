import os

import pandas as pd


def explore_data(data, numerical_vars=None, categorical_vars=None, output_folder='exploratory_analysis'):
    """
    runs a simple exploration of the dataset and writes results to info files, and plots the latitude and longitude data
    :param data: pandas dataframe of the data
    :param categorical_vars: list of str, the categorircal variables in the dataframe
    :param output_folder: str, the folder to write the results to
    """

    print('\n')
    print('variables in data', data.columns)
    print('\n')
    print('number of data points', data.shape[0])
    print('\n')

    # info on missing values
    missing_vals = data.isna().sum()
    with open(f'{output_folder}/data_frame_missing_values.txt', 'w') as fp:
        fp.write(missing_vals.to_string())

    full_desc = data.describe()
    with open(f'{output_folder}/data_frame_summary_numbers.txt', 'w') as fp:
        fp.write(str(full_desc))

    for var in categorical_vars:
        desc = data[var].value_counts()
        with open(f'{output_folder}/{var}_info.txt', 'w') as fp:
            fp.write(f'{len(desc)} number of different values for {var}')
            fp.write('\n')
            fp.write('\n')
            fp.write('values   counts')
            fp.write('\n')
            fp.write(desc.to_string())

    # plot_data = [go.Scatter(x=data['longitude'], y=data['latitude'], mode='markers')]
    # py.offline.plot(go.Figure(data=plot_data),
    #                 filename=f'{output_folder}/longlat.html', auto_open=False)


def main_explore():
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
        os.makedirs(output_folder + '/calls')
        os.makedirs(output_folder + '/leads')
        os.makedirs(output_folder + '/signups')

    explore_data(data=calls,
                 categorical_vars=['Call Outcome', 'Agent'],
                 numerical_vars=['Call Number'],
                 output_folder=output_folder + '/calls')

    explore_data(data=leads,
                 categorical_vars=['Name', 'Phone Number', 'Region', 'Sector', 'Age'],
                 numerical_vars=['Call Number'],
                 output_folder=output_folder + '/leads')

    explore_data(data=signups,
                 categorical_vars=['Lead', 'Approval Decision'],
                 numerical_vars=['Call Number'],
                 output_folder=output_folder + '/signups')


if __name__ == '__main__':
    main_explore()
