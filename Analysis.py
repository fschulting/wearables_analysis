# import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import shapiro
from scipy.stats import wilcoxon   
from scipy import signal
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import pyCompare
import warnings
import re
warnings.simplefilter(action='ignore', category=FutureWarning)

import importlib.metadata

def print_package_versions(*packages):
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            print(f"{package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: not installed")
    
device_colors = {
    'Polar Vantage V3 D733F724': 'tab:blue',
    'Polar Sense D833AF2E': 'tab:red',
    'Polar H10 D6B33A2A': 'tab:green',
    'Fitbit Charge 2': 'tab:orange'
}

subjects = {
    '5f672ff8-5950-48a8-8b5b-b385a0add8f2': 'subject-1',
    '784db1af-fdef-4b1f-8c36-e339b667e6e8': 'subject-2',
    '7b24b3e2-86a3-4857-950f-159697ba8923': 'subject-3',
    '9f2e03a2-0c44-41ea-b93a-e1de91a2a4a0': 'subject-4',
    '2c31dbb8-5985-480d-9a73-b7fcb922a15d': 'subject-5',
    '8d4804ac-eeaa-489a-8033-9a2d4c07ec86': 'subject-6',
    '2b54b23c-11da-45c4-ba6a-9bde6ffcc182': 'subject-7',
    'a37fedae-ba8f-4be9-a686-b9240f3166d7': 'subject-8',
    '81252cbd-852a-4448-acd0-287ffa1de226': 'subject-9',
    '309eb2f0-bb5e-4072-881e-8953444cbd1b': 'subject-10',
    'a59e17c0-2fd3-4e19-9135-9e0afb740047': 'subject-11',
    '7fd42a22-81b6-4f33-95e6-81eacd8ac400': 'subject-12',
    'fd4731fd-b390-467c-977b-48ca733f7d0a': 'subject-13',
    '54414924-a9f9-462e-9aec-b633cdf9ec68': 'fitbit'
}

activities = ['Full', 'Resting', 'Exercise', 'Recovery']

def calculate_freq(dataframe, time_column='Time', device_column='value.deviceName', window_minutes=1):
    dataframe[time_column] = pd.to_datetime(dataframe[time_column])
    
    results = {}
    
    for device in dataframe[device_column].unique():
        device_df = dataframe[dataframe[device_column] == device]

        start_time = device_df[time_column].iloc[0]        
        end_time = start_time + pd.Timedelta(minutes=window_minutes)

        device_df = device_df[(device_df[time_column] >= start_time) & (device_df[time_column] < end_time)]

        frequency = len(device_df) / (window_minutes * 60)
        
        results[device] = frequency
        print(f"{device} freq: {frequency}")

    return results

def get_device_name(device_name):
    # Remove the 'D' followed by alphanumeric characters
    cleaned_name = re.sub(r' D[A-Za-z0-9]+', '', device_name)
    return cleaned_name

def return_subject_timestamp(df, subject, timestamp):
    filtered_df = df.loc[
        (df['key.userId'] == subject) & 
        (df[timestamp].notna()), 
        timestamp
    ]
    return filtered_df.mode().values[0]


def load_files_into_df (filenames, color_map = device_colors, directory = ''):
    dataframes = []
    for filename in filenames:
        df = pd.read_csv(directory + filename, compression='gzip')
        df['filename'] = filename  
        dataframes.append(df)

        if 'fitbit' in filename:
            df['value.deviceName'] = 'Fitbit Charge 2'

    dataframe = pd.concat(dataframes, ignore_index=True)

    # Change 'D733F724' to 'Polar Vantage V3 D733F724'
    dataframe.loc[dataframe['value.deviceName'] == 'D733F724', 'value.deviceName'] = 'Polar Vantage V3 D733F724'
    dataframe.loc[dataframe['value.deviceName'] == 'D833AF2E', 'value.deviceName'] = 'Polar Sense D833AF2E'

    # Timestamp Fitbit is in seconds, while Polar's in ns
    if 'Fitbit Charge 2' in dataframe['value.deviceName'].unique():
        dataframe.loc[dataframe['value.deviceName'] == 'Fitbit Charge 2', 'Time'] = pd.to_datetime(
        dataframe.loc[dataframe['value.deviceName'] == 'Fitbit Charge 2', 'value.time'], unit='s') + pd.Timedelta(hours=2)

        dataframe.loc[dataframe['value.deviceName'] != 'Fitbit Charge 2', 'Time'] = pd.to_datetime(
        dataframe.loc[dataframe['value.deviceName'] != 'Fitbit Charge 2', 'value.time'], unit='ns') + pd.Timedelta(hours=2)

    dataframe = dataframe.sort_values(by='Time')
    dataframe = dataframe.reset_index(drop=True)

    dataframe['Color'] = dataframe['value.deviceName'].map(color_map)

    removed_datapoints_df = remove_datapoints(dataframe, 'Polar Sense D833AF2E', 5)

    return removed_datapoints_df

def shift_time(dataframe, device_name, offset):
    
    shifted = dataframe.copy()
    old_seconds_name = f'old_{device_name}'
    shifted[old_seconds_name] = shifted['Time']

    shifted.loc[shifted['value.deviceName'] == device_name, 'Time'] += offset

    return shifted

def remove_datapoints(df, device, length):

    polar_sense_df = df[df['value.deviceName'] == device].copy()
    all_other_df = df[df['value.deviceName'] != device].copy()

    polar_sense_df['is_stuck'] = False

    rolling_windows = polar_sense_df['value.heartRate'].rolling(window=length, min_periods=length)
    stuck_points = rolling_windows.apply(lambda x: len(set(x)) == 1, raw=True) # Checks if the set has exactly one unique value, meaning all values in the window are the same

    stuck_indices = stuck_points[stuck_points == True].index # Find the indices where the heart rate values are 'stuck'
    for idx in stuck_indices:
        polar_sense_df.loc[idx:idx+length-1, 'is_stuck'] = True # Make is_stuck column with the identified stuck points marked as True

    polar_sense_df = polar_sense_df[polar_sense_df['is_stuck'] == False] # Remove stuck points

    removed_datapoints_df = pd.concat([polar_sense_df, all_other_df], ignore_index=True)

    removed_datapoints_df.drop(columns=['is_stuck'], inplace=True)

    return removed_datapoints_df

def make_plot_per_device(dataframe, color_map = device_colors):

    unique_devices = dataframe['value.deviceName'].unique()
    for device in unique_devices:

        plt.figure(figsize=(12, 8))

        device_data = dataframe[dataframe['value.deviceName'] == device]
        plt.plot(device_data['Time'], device_data['value.heartRate'], color=color_map[device], label=device)
        plt.scatter(device_data['Time'], device_data['value.heartRate'], color='black', s=4, label=device)

        plt.title('Full Heart Rate data')
        plt.xlabel('Seconds')
        plt.ylabel('Heart rate (BPM)')
        plt.legend()
        plt.show()

def ensure_datetime(input_time):
    if not isinstance(input_time, pd.Timestamp):
        time = pd.to_datetime(input_time)
    else:
        time = input_time
    return time

def plot_all_devices(dataframe, start_time, end_time, color_map = device_colors):

    start = ensure_datetime(start_time)
    end = ensure_datetime(end_time)

    dataframe = dataframe[dataframe['Time'] >= start]
    dataframe = dataframe[dataframe['Time'] < end]

    subject = dataframe['key.userId'].unique()[0]

    plt.figure(figsize=(12, 8))

    unique_devices = dataframe['value.deviceName'].unique()
    for device in unique_devices:
        device_data = dataframe[dataframe['value.deviceName'] == device]
        plt.plot(device_data['Time'], device_data['value.heartRate'], color=color_map[device], label=get_device_name(device))
        # plt.scatter(device_data['TimeInMinutes'], device_data['value.heartRate'], color='black', s=4, label=device)

    plt.legend(loc='upper left')

    plt.title('Heart rate of ' + subjects[subject], fontsize=16, fontweight='bold')
    # plt.xlabel('Time (min)', fontsize=14, fontweight='bold')
    plt.ylabel('Heart rate (bpm)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend()
    plt.show()
    
def plot_average_HR(average_HR_df, color_map = device_colors):
    plt.figure(figsize=(12, 8))

    unique_devices = average_HR_df['value.deviceName'].unique()
    for device in unique_devices:
        device_data = average_HR_df[average_HR_df['value.deviceName'] == device]
        plt.plot(device_data['5_sec_time'], device_data['AverageHR_5sec'], color=color_map[device], label=device)
        # plt.scatter(device_data['TimeInMinutes'], device_data['AverageHR'], color='black', s=4, label=device)

    plt.title('Heart rate of', device_data['key.userId'])
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Heart rate (BPM)')
    plt.legend()
    plt.show()    

def make_average_HR_df(dataframe, subject_df):

    df_zero = pd.DataFrame()

    for subject, subject_df in subject_df.groupby('key.userId'):
        activity_df = filter_on_activity(dataframe, subject_df)

        for subject in activity_df['key.userId'].unique():

            subject_df = activity_df[activity_df['key.userId'] == subject]
            
            df_zero_subject = set_time_to_zero(subject_df, 'Fitbit Charge 2', 'Polar H10 D6B33A2A', 'Polar Vantage V3 D733F724')

            df_zero = pd.concat([df_zero, df_zero_subject], ignore_index=True)

    avg_HR_df = df_zero.groupby(['value.deviceName', 'TimeInMinutes'])['value.heartRate'].mean().reset_index()
    avg_HR_df.rename(columns={'value.heartRate': 'AverageHR'}, inplace=True)
    avg_HR_df['Seconds'] = avg_HR_df["TimeInMinutes"].dt.total_seconds()
    avg_HR_df['AverageHR_5sec'] = avg_HR_df['AverageHR'].rolling(window=5, min_periods=1).mean()
    avg_HR_df['5_sec_time'] = pd.NA

    # Assign the first value of every 5th row to 5_sec_time
    for i in range(0, len(avg_HR_df), 5):
        avg_HR_df.loc[i:i+4, '5_sec_time'] = avg_HR_df.loc[i, 'Seconds']

    avg_HR_df['5_sec_time_timedelta'] = pd.to_timedelta(avg_HR_df['5_sec_time'], unit='s')

    avg_HR_df_5sec = [avg_HR_df.iloc[i::4] for i in range(5)][0]
    return avg_HR_df_5sec

def make_mean_HR_10sec(dataframe):
    result_df = pd.DataFrame()

    # Convert Time to total seconds
    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = dataframe['Seconds'] - dataframe['Seconds'].min()
    # Round time to the nearest second
    dataframe['TimeInSeconds'] = pd.to_datetime(dataframe['Time']).dt.round('1s')
    dataframe = dataframe.drop_duplicates(subset=['value.deviceName', 'TimeInSeconds'], keep='first')

    # Group by each device
    for device, device_df in dataframe.groupby('value.deviceName'):
        if device == 'Fitbit Charge 2':  # Exclude 'Fitbit Charge 2' from the calculation
            fitbit_df = device_df.copy()
            fitbit_df['Mean_10s_HR'] = fitbit_df['value.heartRate']
            result_df = pd.concat([result_df, fitbit_df], ignore_index=True)
        else:
            # Calculate the mean heart rate for every 10-second interval
            device_df['Mean_10s_HR'] = device_df['value.heartRate'].rolling(window=10, min_periods=1).mean().shift(-9)
            result_df = pd.concat([result_df, device_df], ignore_index=True)


    return result_df

def set_time_to_zero(dataframe, device1, device2, device3):

    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = (dataframe['Seconds'] - dataframe['Seconds'].min())

    dataframe['TimeInSeconds'] = pd.to_datetime(dataframe['Time']).round('1s')
    dataframe['TimeInSeconds'] = (dataframe['TimeInSeconds'] - dataframe['TimeInSeconds'].min())
    dataframe['TimeInMinutes'] = pd.to_timedelta(dataframe['TimeInSeconds'], unit='s')

    dataframe['TimeString'] = dataframe['TimeInSeconds'].astype(str)

    # Drop 0 
    dataframe = dataframe[dataframe['value.heartRate'] != 0]

    # Drop dupulicated time values
    dataframe = dataframe.drop_duplicates(subset=['value.deviceName', 'TimeInSeconds'], keep='first')
    dataframe_sorted = dataframe.sort_values(by='TimeInSeconds')

    dataframe_sorted = dataframe_sorted.reset_index(drop=True)

    activities = ['Resting', 'Exercise', 'Recovery']

    common_seconds_set = set()

    for activity in activities:
        device1_activity = dataframe_sorted[(dataframe_sorted['Activity'] == activity) & (dataframe_sorted['value.deviceName'] == device1)]
        device2_activity = dataframe_sorted[(dataframe_sorted['Activity'] == activity) & (dataframe_sorted['value.deviceName'] == device2)]
        device3_activity = dataframe_sorted[(dataframe_sorted['Activity'] == activity) & (dataframe_sorted['value.deviceName'] == device3)]

        common_seconds_activity = set(set(device1_activity['TimeInSeconds']).intersection(set(device2_activity['TimeInSeconds']))).intersection(set(device3_activity['TimeInSeconds']))
        common_seconds_set.update(common_seconds_activity)

    common_seconds = pd.Series(list(common_seconds_set))

    # Filter the dataframe based on common seconds
    dataframe_filtered = dataframe_sorted[dataframe_sorted['TimeInSeconds'].isin(common_seconds)]
    dataframe_filtered.sort_values(by='TimeInSeconds')

    return dataframe_filtered

def filter_common_timepoints_H10_sense(dataframe):
    # Convert Time to total seconds
    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = dataframe['Seconds'] - dataframe['Seconds'].min()

    # Round time to the nearest second
    dataframe['TimeInSeconds'] = dataframe['Time'].dt.round('1s')

    # Drop rows with heart rate of 0
    dataframe = dataframe[dataframe['value.heartRate'] != 0]

    # Drop duplicated time values for each device
    dataframe = dataframe.drop_duplicates(subset=['value.deviceName', 'TimeInSeconds'], keep='first')

    # Sort the dataframe by time
    dataframe_sorted = dataframe.sort_values(by='TimeInSeconds').reset_index(drop=True)

    # Filter dataframes for each device
    device_dfs = {
        'H10': dataframe_sorted[dataframe_sorted['value.deviceName'] == 'Polar H10 D6B33A2A'],
        'Sense': dataframe_sorted[dataframe_sorted['value.deviceName'] == 'Polar Sense D833AF2E'],
    }

    # Print information for each device
    print('Before filtering on common timepoints')
    for name, df in dataframe_sorted.groupby('value.deviceName'):
        print(f"Device: {name}")
        print(f"Total records (without Unknown): {len(df) - len(df[df['Activity'] == 'Unknown'])}")
        for activity in ['Resting', 'Exercise', 'Recovery', 'Unknown']:
            print(f"{activity}: {len(df[df['Activity'] == activity])}")
        print()

    # Function to find common timepoints for a given activity
    def find_common_timepoints(activity):
        timepoints_sets = [set(df[df['Activity'] == activity]['TimeInSeconds']) for df in device_dfs.values()]
        common_timepoints = set.intersection(*timepoints_sets)
        filtered_df = dataframe_sorted[dataframe_sorted['TimeInSeconds'].isin(common_timepoints)]
        filtered_df['Activity'] = activity
        return filtered_df

    # Get filtered dataframes for each activity
    dataframe_filtered_resting = find_common_timepoints('Resting')
    dataframe_filtered_exercise = find_common_timepoints('Exercise')
    dataframe_filtered_recovery = find_common_timepoints('Recovery')

    # Concatenate all filtered dataframes
    dataframe_filtered = pd.concat([dataframe_filtered_resting, dataframe_filtered_exercise, dataframe_filtered_recovery])

    # Sort the concatenated dataframe by time
    dataframe_filtered = dataframe_filtered.sort_values(by='TimeInSeconds').reset_index(drop=True)

    return dataframe_filtered


def filter_common_timepoints_all_devices(dataframe, device1, device2, device3):
    # Convert Time to total seconds
    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = dataframe['Seconds'] - dataframe['Seconds'].min()

    # Round time to the nearest second
    dataframe['TimeInSeconds'] = dataframe['Time'].dt.round('1s')

    # Drop rows with heart rate of 0
    dataframe = dataframe[dataframe['value.heartRate'] != 0]

    # Drop duplicated time values for each device
    dataframe = dataframe.drop_duplicates(subset=['value.deviceName', 'TimeInSeconds'], keep='first')

    # Sort the dataframe by time
    dataframe_sorted = dataframe.sort_values(by='TimeInSeconds').reset_index(drop=True)

    # Filter dataframes for each device
    device_dfs = {
        device1: dataframe_sorted[dataframe_sorted['value.deviceName'] == device1],
        device2: dataframe_sorted[dataframe_sorted['value.deviceName'] == device2],
        device3: dataframe_sorted[dataframe_sorted['value.deviceName'] == device3]
    }

    # Print information for each device
    print('Before filtering on common timepoints')
    for name, df in dataframe_sorted.groupby('value.deviceName'):
        print(f"Device: {name}")
        print(f"Total records (without Unknown): {len(df) - len(df[df['Activity'] == 'Unknown'])}")
        for activity in ['Resting', 'Exercise', 'Recovery', 'Unknown']:
            print(f"{activity}: {len(df[df['Activity'] == activity])}")
        print()

    # Function to find common timepoints for a given activity
    def find_common_timepoints(activity):
        timepoints_sets = [set(df[df['Activity'] == activity]['TimeInSeconds']) for df in device_dfs.values()]
        common_timepoints = set.intersection(*timepoints_sets)
        filtered_df = dataframe_sorted[dataframe_sorted['TimeInSeconds'].isin(common_timepoints)]
        filtered_df['Activity'] = activity
        return filtered_df

    # Get filtered dataframes for each activity
    dataframe_filtered_resting = find_common_timepoints('Resting')
    dataframe_filtered_exercise = find_common_timepoints('Exercise')
    dataframe_filtered_recovery = find_common_timepoints('Recovery')

    # Concatenate all filtered dataframes
    dataframe_filtered = pd.concat([dataframe_filtered_resting, dataframe_filtered_exercise, dataframe_filtered_recovery])

    # Sort the concatenated dataframe by time
    dataframe_filtered = dataframe_filtered.sort_values(by='TimeInSeconds').reset_index(drop=True)

    # Print information for each device
    print('After filtering on common timepoints')
    for name, df in dataframe_filtered.groupby('value.deviceName'):
        print(f"Device: {name}")
        print(f"Total records: {len(df)}")
        for activity in ['Resting', 'Exercise', 'Recovery', 'Unknown']:
            print(f"{activity}: {len(df[df['Activity'] == activity])}")
        print()

    return dataframe_filtered


def filter_common_timepoints(dataframe, device1, device2):

    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = (dataframe['Seconds'] - dataframe['Seconds'].min())

    dataframe['TimeInSeconds'] = pd.to_datetime(dataframe['Time']).round('1s')

    # Drop 0 
    dataframe = dataframe[dataframe['value.heartRate'] != 0]

    # Drop dupulicated time values
    dataframe = dataframe.drop_duplicates(subset=['value.deviceName', 'TimeInSeconds'], keep='first')
    dataframe_sorted = dataframe.sort_values(by='TimeInSeconds')

    dataframe_sorted = dataframe_sorted.reset_index(drop=True)

    activities = ['Resting', 'Exercise', 'Recovery']

    common_seconds_set = set()

    for activity in activities:
        device1_activity = dataframe_sorted[(dataframe_sorted['Activity'] == activity) & (dataframe_sorted['value.deviceName'] == device1)]
        device2_activity = dataframe_sorted[(dataframe_sorted['Activity'] == activity) & (dataframe_sorted['value.deviceName'] == device2)]

        common_seconds_activity = set(device1_activity['TimeInSeconds']).intersection(set(device2_activity['TimeInSeconds']))
        common_seconds_set.update(common_seconds_activity)

    common_seconds = pd.Series(list(common_seconds_set))

    # Filter the dataframe based on common seconds
    dataframe_filtered = dataframe_sorted[dataframe_sorted['TimeInSeconds'].isin(common_seconds)]
    dataframe.sort_values(by='TimeInSeconds')

    return dataframe_filtered


def calculate_mean_10s(dataframe):

    dataframe['Seconds'] = dataframe['Time'].dt.hour * 3600 + dataframe['Time'].dt.minute * 60 + dataframe['Time'].dt.second
    dataframe['Seconds'] = (dataframe['Seconds'] - dataframe['Seconds'].min())

    return dataframe


def check_measurements_per_device_pair(dataframe, device1, device2):
    n_measurements_device1 = dataframe[dataframe['value.deviceName'] == device1].shape[0]
    n_measurements_device2 = dataframe[dataframe['value.deviceName'] == device2].shape[0]

    n_measurements_resting = dataframe[(dataframe['value.deviceName'] == device2) & (dataframe['Activity'] == 'Resting')].shape[0]
    n_measurements_exercise = dataframe[(dataframe['value.deviceName'] == device2) & (dataframe['Activity'] == 'Exercise')].shape[0]
    n_measurements_recovery = dataframe[(dataframe['value.deviceName'] == device2) & (dataframe['Activity'] == 'Recovery')].shape[0]

    if n_measurements_device1 == n_measurements_device2:
        print('Both devices have', n_measurements_device1, 'measurements  - ', device1, device2)
        print('Resting', n_measurements_resting)
        print('Exercise', n_measurements_exercise)
        print('Recovery', n_measurements_recovery)

        return True
    else:
        print(f'Measurements mismatch: {device1} has {n_measurements_device1}, {device2} has {n_measurements_device2}')
        return False


def filter_on_activity(dataframe, subject_df):

    unique_subjects = subject_df['key.userId'].unique()

    merged_df = pd.merge(dataframe, subject_df, how='left', on='key.userId')

    filtered_df = pd.DataFrame()

    for subject in unique_subjects:
        if subject not in merged_df['key.userId'].values:
            print('You are missing data from', subject)
            continue
        
        start_time = merged_df.loc[merged_df['key.userId'] == subject, 'start'].values[0]
        resting_start = merged_df.loc[merged_df['key.userId'] == subject, 'resting_start'].values[0]
        resting_end = merged_df.loc[merged_df['key.userId'] == subject, 'resting_end'].values[0]
        exercise_start = merged_df.loc[merged_df['key.userId'] == subject, 'exercise_start'].values[0]
        exercise_end = merged_df.loc[merged_df['key.userId'] == subject, 'exercise_end'].values[0]
        recovery_start = merged_df.loc[merged_df['key.userId'] == subject, 'recovery_start'].values[0]
        recovery_end = merged_df.loc[merged_df['key.userId'] == subject, 'recovery_end'].values[0]

        filtered_subject_df = merged_df[(merged_df['Time'] > start_time)]
        filtered_subject_df = filtered_subject_df[filtered_subject_df['Time'] < recovery_end]

        if 'Fitbit Charge 2' in merged_df['value.deviceName'].values:
            filtered_subject_df['key.userId'] = subject
            filtered_subject_df['Activity'] = 'Unknown'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > resting_start) & (filtered_subject_df['Time'] < resting_end), 'Activity'] = 'Resting'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > exercise_start) & (filtered_subject_df['Time'] < exercise_end), 'Activity'] = 'Exercise'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > recovery_start) & (filtered_subject_df['Time'] < recovery_end), 'Activity'] = 'Recovery'

        filtered_df = pd.concat([filtered_df, filtered_subject_df])

    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

def old_filter_on_activity(dataframe, subject_df):

    unique_subjects = subject_df['key.userId'].unique()
    merged_df = pd.merge(dataframe, subject_df, how='left', on='key.userId')
        
    filtered_df = pd.DataFrame()

    for subject in unique_subjects:
        if subject not in merged_df['key.userId'].values:
            print('You are missing data from', subject)
            continue

        start_time = merged_df.loc[merged_df['key.userId'] == subject, 'start'].values[0]
        resting_start = merged_df.loc[merged_df['key.userId'] == subject, 'resting_start'].values[0]
        resting_end = merged_df.loc[merged_df['key.userId'] == subject, 'resting_end'].values[0]
        exercise_start = merged_df.loc[merged_df['key.userId'] == subject, 'exercise_start'].values[0]
        exercise_end = merged_df.loc[merged_df['key.userId'] == subject, 'exercise_end'].values[0]
        recovery_start = merged_df.loc[merged_df['key.userId'] == subject, 'recovery_start'].values[0]
        recovery_end = merged_df.loc[merged_df['key.userId'] == subject, 'recovery_end'].values[0]

        filtered_subject_df = merged_df[(merged_df['Time'] > start_time)]
        filtered_subject_df = filtered_subject_df[filtered_subject_df['Time'] < recovery_end]
        
        if 'Fitbit Charge 2' in merged_df['value.deviceName'].values:
            filtered_subject_df['key.userId'] = subject

            filtered_subject_df['Activity'] = 'Unknown'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > resting_start) & (filtered_subject_df['Time'] < resting_end), 'Activity'] = 'Resting'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > exercise_start) & (filtered_subject_df['Time'] < exercise_end), 'Activity'] = 'Exercise'
            filtered_subject_df.loc[(filtered_subject_df['Time'] > recovery_start) & (filtered_subject_df['Time'] < recovery_end), 'Activity'] = 'Recovery'

        filtered_df = pd.concat([filtered_df, filtered_subject_df])

    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

def make_plot_per_activity(dataframe, subject):

    activity_df = dataframe.copy()
    subject_df = activity_df[activity_df['key.userId'] == subject]

    unique_activities = subject_df['Activity'].unique()
    for activity in unique_activities:
        plt.figure(figsize=(12, 8))

        activity_data = subject_df[subject_df['Activity'] == activity]
        unique_devices = subject_df['value.deviceName'].unique()
        
        for device in unique_devices:
            device_data = activity_data[activity_data['value.deviceName'] == device]
            plt.plot(device_data['Time'], device_data['value.heartRate'], color=device_colors[device], label=device)
            plt.scatter(device_data['Time'], device_data['value.heartRate'], color='black', s=4, label=device)


        plt.title('Heart rate of ' + subject + ' during ' + activity)
        plt.xlabel('Time')
        plt.ylabel('Heart rate (BPM)')
        plt.legend()
        plt.show()

def split_HR_data_per_activity(df, device_name):

    device_data = df[df['value.deviceName'] == device_name]
    device_array = device_data[['value.heartRate', 'Seconds', 'Activity']].to_numpy()

    full = device_array[:, 2] != 'Unknown'
    resting_filter = device_array[:, 2] == 'Resting'
    exercise_filter = device_array[:, 2] == 'Exercise'
    recovery_filter = device_array[:, 2] == 'Recovery'

    HR_all = device_array[full, 0].astype(int)
    HR_resting = device_array[resting_filter, 0].astype(int)
    HR_exercise = device_array[exercise_filter, 0].astype(int)
    HR_recovery = device_array[recovery_filter, 0].astype(int)

    return HR_all, HR_resting, HR_exercise, HR_recovery

def split_HR_data_per_activity_time(df, device_name):

    device_data = df[df['value.deviceName'] == device_name]
    device_array = device_data[['value.heartRate', 'Time', 'Activity']].to_numpy()

    full = device_array[:, 2] != 'Unknown'
    resting_filter = device_array[:, 2] == 'Resting'
    exercise_filter = device_array[:, 2] == 'Exercise'
    recovery_filter = device_array[:, 2] == 'Recovery'

    HR_all = device_array[full, 0].astype(int)
    HR_resting = device_array[resting_filter, 0].astype(int)
    HR_exercise = device_array[exercise_filter, 0].astype(int)
    HR_recovery = device_array[recovery_filter, 0].astype(int)

    return HR_all, HR_resting, HR_exercise, HR_recovery


def paired_T_test(df, device1_name, device2_name, subject_df):

    filtered_df_device1_device2 = filter_on_activity(df, subject_df)

    filtered_df_device1_device2 = filter_common_timepoints(filtered_df_device1_device2, device1_name, device2_name)

    if not check_measurements_per_device_pair(filtered_df_device1_device2, device1_name, device2_name):
        print('Something goes wrong as number of measurements between devices does not match')

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device2_name)

    try:
        paired_T_test, p = stats.ttest_rel(device1_HR, device2_HR)
        paired_T_test_resting, p_resting = stats.ttest_rel(device1_HR_resting, device2_HR_resting)
        paired_T_test_exercise, p_exercise = stats.ttest_rel(device1_HR_exercise, device2_HR_exercise)
        paired_T_test_recovery, p_recovery = stats.ttest_rel(device1_HR_recovery, device2_HR_recovery)
        
        print('Paired T-test results for ', device1_name, device2_name)
        print('All data:', paired_T_test, p)
        print('Resting data:', paired_T_test_resting, p_resting)
        print('Exercise data:', paired_T_test_exercise, p_exercise)
        print('Recovery data:', paired_T_test_recovery, p_recovery)
        print('')

    except ValueError as e:
        print('Error in T-test:', e)


def Pearson_correlation(df, device1_name, device2_name, subject_df):

    filtered_df_device1_device2 = filter_on_activity(df, subject_df)
    filtered_df_device1_device2 = filter_common_timepoints(filtered_df_device1_device2, device1_name, device2_name)

    if not check_measurements_per_device_pair(filtered_df_device1_device2, device1_name, device2_name):
        print('Something goes wrong as number of measurements between devices does not match')

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device2_name)

    try:
        cor_coef, p = pearsonr(device1_HR, device2_HR)
        cor_coef_resting, p_resting = pearsonr(device1_HR_resting, device2_HR_resting)
        cor_coef_exercise, p_exercise = pearsonr(device1_HR_exercise, device2_HR_exercise)
        cor_coef_recovery, p_recovery = pearsonr(device1_HR_recovery, device2_HR_recovery)
        
        print('Pearson results for ', device1_name, device2_name)
        print('All data:', cor_coef, p)
        print('Resting data:', cor_coef_resting, p_resting)
        print('Exercise data:', cor_coef_exercise, p_exercise)
        print('Recovery data:', cor_coef_recovery, p_recovery)
        print('')
    
    except ValueError as e:
        print('Error in Pearson:', e)


def shapiro_wilk_test(df, device1_name, device2_name, device3_name, subject_df):

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)
    device3_HR, device3_HR_resting, device3_HR_exercise, device3_HR_recovery = split_HR_data_per_activity(df, device3_name)


    try:
        stat_1, p_value_1 = shapiro(device1_HR)
        stat_1_resting, p_value_1_resting = shapiro(device1_HR_resting)
        stat_1_exercise, p_value_1_exercise = shapiro(device1_HR_exercise)
        stat_1_recovery, p_value_1_recovery = shapiro(device1_HR_recovery)

        stat_2, p_value_2 = shapiro(device2_HR)
        stat_2_resting, p_value_2_resting = shapiro(device2_HR_resting)
        stat_2_exercise, p_value_2_exercise = shapiro(device2_HR_exercise)
        stat_2_recovery, p_value_2_recovery = shapiro(device2_HR_recovery)

        stat_3, p_value_3 = shapiro(device3_HR)
        stat_3_resting, p_value_3_resting = shapiro(device3_HR_resting)
        stat_3_exercise, p_value_3_exercise = shapiro(device3_HR_exercise)
        stat_3_recovery, p_value_3_recovery = shapiro(device3_HR_recovery)


        print(device1_name, 'Shapiro-Wilk Test Statistic:', stat_1, stat_1_resting, stat_1_exercise, stat_1_recovery)
        print('p-value:', p_value_1, p_value_1_resting, p_value_1_exercise, p_value_1_recovery)
        if p_value_1 > 0.05:
            print(device1_name,"is normally distributed (full)")
        else:
            print(device1_name, "is not normally distributed (full)")
        print('')
        print(device2_name, 'Shapiro-Wilk Test Statistic:', stat_2, stat_2_resting, stat_2_exercise, stat_2_recovery)
        print('p-value:', p_value_2, p_value_2_resting, p_value_2_exercise, p_value_2_recovery)
        if p_value_2 > 0.05:
            print(device2_name,"is normally distributed (full)")
        else:
            print(device2_name, "is not normally distributed (full)")
        print('')
        print(device3_name, 'Shapiro-Wilk Test Statistic:', stat_3, stat_3_resting, stat_3_exercise, stat_3_recovery)
        print('p-value:', p_value_3, p_value_3_resting, p_value_3_exercise, p_value_3_recovery)
        if p_value_3 > 0.05:
            print(device3_name,"is normally distributed (full)")
        else:
            print(device3_name, "is not normally distributed (full)")

    except ValueError as e:
        print('Error:', e)

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator

def test_per_subject_per_devicepair(test, df, device1_name, device2_name, subject_df):
    
    scores = []

    for subject_id in subject_df['key.userId'].unique():

        subject_data = df[df['key.userId'] == subject_id]

        print(subject_id)
        result, result_resting, result_exercise, result_recovery = test(subject_data, device1_name, device2_name, subject_df)
        scores.append([result, result_resting, result_exercise, result_recovery])

    print("")
    print(scores)
    print('Full', np.mean([row[0] for row in scores]), np.std([row[0] for row in scores]), 'min', np.min([row[0] for row in scores]), 'max', np.max([row[0] for row in scores]))    
    print('Rest', np.mean([row[1] for row in scores]), np.std([row[1] for row in scores]), 'min', np.min([row[1] for row in scores]), 'max', np.max([row[1] for row in scores]))    
    print('Exercise', np.mean([row[2] for row in scores]), np.std([row[2] for row in scores]), 'min', np.min([row[2] for row in scores]), 'max', np.max([row[2] for row in scores]))    
    print('Recovery', np.mean([row[3] for row in scores]), np.std([row[3] for row in scores]), 'min', np.min([row[3] for row in scores]), 'max', np.max([row[3] for row in scores]))    


def ccc_per_devicepair(df, device1_name, device2_name, subject_df):

    # Split HR data into different activities for both devices
    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)

    try:
        ccc  = concordance_correlation_coefficient(device1_HR, device2_HR)
        ccc_resting  = concordance_correlation_coefficient(device1_HR_resting, device2_HR_resting)
        ccc_exercise  = concordance_correlation_coefficient(device1_HR_exercise, device2_HR_exercise)
        ccc_recovery = concordance_correlation_coefficient(device1_HR_recovery, device2_HR_recovery)
        
        print('concordance_correlation_coefficient results for ', device1_name, device2_name)
        print('All data:', ccc, 'n = ', len(device1_HR))
        print('Resting data:', ccc_resting, 'n = ', len(device1_HR_resting))
        print('Exercise data:', ccc_exercise, 'n = ', len(device1_HR_exercise))
        print('Recovery data:', ccc_recovery, 'n = ', len(device1_HR_recovery))
        print('')
    
    except ValueError as e:
        print('Error:', e)
        print()
        print('Are length equal?', len(device1_HR) == len(device2_HR))  
    
    return ccc, ccc_resting, ccc_exercise, ccc_recovery


def spearman_per_devicepair(df, device1_name, device2_name, subject_df):

    # Split HR data into different activities for both devices
    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)

    try:
        cor_coef, p = spearmanr(device1_HR, device2_HR)
        cor_coef_resting, p_resting = spearmanr(device1_HR_resting, device2_HR_resting)
        cor_coef_exercise, p_exercise = spearmanr(device1_HR_exercise, device2_HR_exercise)
        cor_coef_recovery, p_recovery = spearmanr(device1_HR_recovery, device2_HR_recovery)
        
        print('Spearman results for ', device1_name, device2_name)
        print('All data:', cor_coef, p)
        print('Resting data:', cor_coef_resting, p_resting)
        print('Exercise data:', cor_coef_exercise, p_exercise)
        print('Recovery data:', cor_coef_recovery, p_recovery)
        print('')
    
    except ValueError as e:
        print('Error:', e)
        print()
        print('Are length equal?', len(device1_HR) == len(device2_HR))  

    return cor_coef, cor_coef_resting, cor_coef_exercise, cor_coef_recovery

def ccc_per_subject_per_devicepair(df, device1_name, device2_name, subject_df):

    
    for subject_id in subject_df['key.userId'].unique():

        subject_data = df[df['key.userId'] == subject_id]
    
        # Split HR data into different activities for both devices
        device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(subject_data, device1_name)
        device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(subject_data, device2_name)

        try:
            # Perform the Wilcoxon test on the HR data from both devices
            ccc  = concordance_correlation_coefficient(device1_HR, device2_HR)
            ccc_resting  = concordance_correlation_coefficient(device1_HR_resting, device2_HR_resting)
            ccc_exercise  = concordance_correlation_coefficient(device1_HR_exercise, device2_HR_exercise)
            ccc_recovery = concordance_correlation_coefficient(device1_HR_recovery, device2_HR_recovery)
            
            print(subject_id)
            print('concordance_correlation_coefficient results for ', device1_name, device2_name)
            print('All data:', ccc)
            print('Resting data:', ccc_resting)
            print('Exercise data:', ccc_exercise)
            print('Recovery data:', ccc_recovery)
            print('')
        
        except ValueError as e:
            print(subject_id, 'Error:', e)
            print('')
            print('Are length equal?', len(device1_HR) == len(device2_HR))  
        
    
def wilcoxon_test_per_subject_per_devicepair(df, device1_name, device2_name, subject_df):

    
    for subject_id in subject_df['key.userId'].unique():

        subject_data = df[df['key.userId'] == subject_id]
    
        # Split HR data into different activities for both devices
        device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(subject_data, device1_name)
        device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(subject_data, device2_name)

        try:
            # Perform the Wilcoxon test on the HR data from both devices
            stat_wilcoxon, p_wilcoxon = wilcoxon(device1_HR, device2_HR)
            stat_wilcoxon_resting, p_wilcoxon_resting = wilcoxon(device1_HR_resting, device2_HR_resting)
            stat_wilcoxon_exercise, p_wilcoxon_exercise = wilcoxon(device1_HR_exercise, device2_HR_exercise)
            stat_wilcoxon_recovery, p_wilcoxon_recovery = wilcoxon(device1_HR_recovery, device2_HR_recovery)
            
            print(subject_id)
            print('Wilicoxon results for ', device1_name, device2_name)
            print('All data:', stat_wilcoxon, p_wilcoxon)
            # print('Resting data:', stat_wilcoxon_resting, p_wilcoxon_resting)
            # print('Exercise data:', stat_wilcoxon_exercise, p_wilcoxon_exercise)
            # print('Recovery data:', stat_wilcoxon_recovery, p_wilcoxon_recovery)
            print('')
        
        except ValueError as e:
            print(subject_id, 'Error:', e)
            print('')
            print('Are length equal?', len(device1_HR) == len(device2_HR))  


def wilcoxon_test(df, device1_name, device2_name, subject_df):
    # Filter data for relevant activity
    filtered_df_device1_device2 = filter_on_activity(df, subject_df)
    filtered_df_device1_device2 = filter_common_timepoints(filtered_df_device1_device2, device1_name, device2_name)

    # Check if the number of measurements matches between the two devices
    if not check_measurements_per_device_pair(filtered_df_device1_device2, device1_name, device2_name):
        print('Something goes wrong as the number of measurements between devices does not match')
        return

    # Split HR data into different activities for both devices
    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device2_name)

    try:
        # Perform the Wilcoxon test on the HR data from both devices
        stat_wilcoxon, p_wilcoxon = wilcoxon(device1_HR, device2_HR)
        stat_wilcoxon_resting, p_wilcoxon_resting = wilcoxon(device1_HR_resting, device2_HR_resting)
        stat_wilcoxon_exercise, p_wilcoxon_exercise = wilcoxon(device1_HR_exercise, device2_HR_exercise)
        stat_wilcoxon_recovery, p_wilcoxon_recovery = wilcoxon(device1_HR_recovery, device2_HR_recovery)
        
        print('Wilicoxon results for ', device1_name, device2_name)
        print('All data:', stat_wilcoxon, p_wilcoxon)
        print('Resting data:', stat_wilcoxon_resting, p_wilcoxon_resting)
        print('Exercise data:', stat_wilcoxon_exercise, p_wilcoxon_exercise)
        print('Recovery data:', stat_wilcoxon_recovery, p_wilcoxon_recovery)
        print('')
    
    except ValueError as e:
        print('Error:', e)


def Spearman_correlation(df, device1_name, device2_name, subject_df):

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)

    try:
        cor_coef, p = spearmanr(device1_HR, device2_HR)
        cor_coef_resting, p_resting = spearmanr(device1_HR_resting, device2_HR_resting)
        cor_coef_exercise, p_exercise = spearmanr(device1_HR_exercise, device2_HR_exercise)
        cor_coef_recovery, p_recovery = spearmanr(device1_HR_recovery, device2_HR_recovery)
        
        print('Spearman results for ', device1_name, device2_name)
        print('All data:', cor_coef, p)
        print('Resting data:', cor_coef_resting, p_resting)
        print('Exercise data:', cor_coef_exercise, p_exercise)
        print('Recovery data:', cor_coef_recovery, p_recovery)
        print('')
    
    except ValueError as e:
        print('Error in Spearman:', e)
    
    return cor_coef, cor_coef_resting, cor_coef_exercise, cor_coef_recovery


def RMSE(df, device1_name, device2_name, subject_df):

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)

    try:
        rmse = sqrt(mean_squared_error(device1_HR, device2_HR))
        rmse_resting = sqrt(mean_squared_error(device1_HR_resting, device2_HR_resting))
        rmse_exercise = sqrt(mean_squared_error(device1_HR_exercise, device2_HR_exercise))
        rmse_recovery = sqrt(mean_squared_error(device1_HR_recovery, device2_HR_recovery))
        
        print('RMSE results for ', device1_name, device2_name)
        print('All data:', rmse)
        print('Resting data:', rmse_resting)
        print('Exercise data:', rmse_exercise)
        print('Recovery data:', rmse_recovery)
        print('')
    
    except ValueError as e:
        print('Error in RMSE:', e)

def calculate_NRMSE(observed, predicted):
    rmse = sqrt(mean_squared_error(observed, predicted))
    normalized_rmse = rmse / (observed.max() - observed.min())
    return normalized_rmse

def NRMSE(df, device1_name, device2_name, subject_df):

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(df, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(df, device2_name)

    try:
        rmse = calculate_NRMSE(device1_HR, device2_HR)
        rmse_resting = calculate_NRMSE(device1_HR_resting, device2_HR_resting)
        rmse_exercise = calculate_NRMSE(device1_HR_exercise, device2_HR_exercise)
        rmse_recovery = calculate_NRMSE(device1_HR_recovery, device2_HR_recovery)
        
        print('NRMSE results for ', device1_name, device2_name)
        print('All data:', rmse)
        print('Resting data:', rmse_resting)
        print('Exercise data:', rmse_exercise)
        print('Recovery data:', rmse_recovery)
        print('')
    
    except ValueError as e:
        print('Error in RMSE:', e)



def make_plot_means(df, subject_df, color_map=device_colors):
    
    polar_H10_heartRate = split_HR_data_per_activity_time(df, 'Polar H10 D6B33A2A')[0]
    polar_V3_heartRate = split_HR_data_per_activity_time(df, 'Polar Vantage V3 D733F724')[0]
    polar_Sense_heartRate = split_HR_data_per_activity_time(df, 'Polar Sense D833AF2E')[0]
    fitbit_heartRate = split_HR_data_per_activity_time(df, 'Fitbit Charge 2')[0]

    data = {
        'HeartRate': np.concatenate([polar_H10_heartRate, polar_V3_heartRate, polar_Sense_heartRate, fitbit_heartRate]),
        'Device': ['Polar H10'] * len(polar_H10_heartRate) + 
                ['Polar Vantage V3'] * len(polar_V3_heartRate) + 
                ['Polar Verity Sense'] * len(polar_Sense_heartRate) +
                ['Fitbit Charge 2'] * len(fitbit_heartRate)
    }
    df_heartRate = pd.DataFrame(data)

    device_colors = {
        'Polar Vantage V3': 'tab:blue',
        'Polar Verity Sense': 'tab:red',
        'Polar H10': 'tab:green',
        'Fitbit Charge 2': 'tab:orange'
    }

    # Extract the colors in the same order as the devices in the DataFrame
    unique_devices = df_heartRate['Device'].unique()
    palette = [device_colors[device] for device in unique_devices]

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Device', y='HeartRate', data=df_heartRate, palette=palette)
    # plt.title('Comparison of Heart Rate Between Devices')
    plt.xlabel('', fontsize=14, fontweight='bold')
    plt.ylabel('Heart Rate (bpm)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.show()

def make_scattterplot_H10(df, activity, subject_df):

    fitbit_HR, fitbit_HR_resting, fitbit_HR_exercise, fitbit_HR_recovery = split_HR_data_per_activity(df, 'Fitbit Charge 2')
    fitbitH10_HR, fitbitH10_HR_resting, fitbitH10_HR_exercise, fitbitH10_HR_recovery = split_HR_data_per_activity(df, 'Polar H10 D6B33A2A')
    vantagev3_HR, vantagev3_HR_resting, vantagev3_HR_exercise, vantagev3_HR_recovery = split_HR_data_per_activity(df, 'Polar Vantage V3 D733F724')
    h10_HR, h10_HR_resting, h10_HR_exercise, h10_HR_recovery = split_HR_data_per_activity(df, 'Polar H10 D6B33A2A')

    activity_map = {
        'Full': (fitbit_HR, vantagev3_HR, h10_HR, fitbitH10_HR),
        'Resting': (fitbit_HR_resting, vantagev3_HR_resting, h10_HR_resting, fitbitH10_HR_resting),
        'Exercise': (fitbit_HR_exercise, vantagev3_HR_exercise, h10_HR_exercise, fitbitH10_HR_exercise),
        'Recovery': (fitbit_HR_recovery, vantagev3_HR_recovery, h10_HR_recovery, fitbitH10_HR_recovery)
    }

    if activity not in activity_map:
        raise ValueError(f"Invalid activity: {activity}. Must be one of {list(activity_map.keys())}.")

    data_fitbit, data_vantagev3, data_h10, data_fitbitH10 = activity_map[activity]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x=data_h10, y=data_fitbit, label=f'{activity} Activity', color='tab:orange', scatter_kws={'s': 4})
    sns.regplot(x=data_h10, y=data_vantagev3, label=f'{activity} Activity', color='tab:blue', scatter_kws={'s': 4})

    # plt.title(f'Heart Rate Comparison During {activity}')
    # plt.title(f'Correlation')
    plt.ylabel(f'Polar H10 HR measurements (bpm)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Polar Vantage V3 and Fitbit Charge 2 HR measurements (bpm)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def make_scatterplot_fitbit(df, subject_df):

    fitbit_HR, fitbit_HR_resting, fitbit_HR_exercise, fitbit_HR_recovery = split_HR_data_per_activity(df, 'Fitbit Charge 2')
    vantagev3_HR, vantagev3_HR_resting, vantagev3_HR_exercise, vantagev3_HR_recovery = split_HR_data_per_activity(df, 'Polar Vantage V3 D733F724')
    h10_HR, h10_HR_resting, h10_HR_exercise, h10_HR_recovery = split_HR_data_per_activity(df, 'Polar H10 D6B33A2A')

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x=h10_HR, y=fitbit_HR, label=f'hoi', color='tab:green', scatter_kws={'s': 4})
    sns.regplot(x=vantagev3_HR, y=fitbit_HR, label=f'hoi', color='tab:blue', scatter_kws={'s': 4})


    # plt.title(f'Heart Rate Comparison During {activity}')
    # plt.title(f'Correlation')
    plt.ylabel(f'Fitbit Charge 2 HR measurements (bpm)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Polar Vantage V3 and Polar H10 HR measurements (bpm)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def make_df_bland_altman_per_device(df, device_name):

    device_data = df[df['value.deviceName'] == device_name]
    device_array = device_data[['value.heartRate', 'TimeInSeconds', 'Activity']].to_numpy()

    full = device_array[:, 2] != 'Unknown'
    resting_filter = device_array[:, 2] == 'Resting'
    exercise_filter = device_array[:, 2] == 'Exercise'
    recovery_filter = device_array[:, 2] == 'Recovery'

    HR_all = device_array[full, 0].astype(int)
    HR_resting = device_array[resting_filter, 0].astype(int)
    HR_exercise = device_array[exercise_filter, 0].astype(int)
    HR_recovery = device_array[recovery_filter, 0].astype(int)

    df_all = pd.DataFrame({'value.heartRate': HR_all, 'Activity': 'All'})
    df_resting = pd.DataFrame({'value.heartRate': HR_resting, 'Activity': 'Resting'})
    df_exercise = pd.DataFrame({'value.heartRate': HR_exercise, 'Activity': 'Exercise'})
    df_recovery = pd.DataFrame({'value.heartRate': HR_recovery, 'Activity': 'Recovery'})

    return pd.concat([df_resting, df_exercise, df_recovery], ignore_index=True)

def data_for_bland_altman(df, subject_df, device1_name, device2_name, percentage=False):
    
    device1_HR = make_df_bland_altman_per_device(df, device1_name)
    device2_HR = make_df_bland_altman_per_device(df, device2_name)

    if isinstance(device1_HR, pd.DataFrame) and isinstance(device2_HR, pd.DataFrame):
        merged_HR = pd.concat([device1_HR['value.heartRate'].reset_index(drop=True), device2_HR.reset_index(drop=True)], axis=1)

        merged_HR.columns = [device1_name, device2_name, 'Activity']
        return merged_HR
    else:
        raise TypeError("Both device1_HR and device2_HR must be DataFrames")


def bland_altman_plot_activity(df, device1, device2, full_data = False):

    df['mean'] = df[[device1, device2]].mean(axis=1)
    df['difference'] = df[device1] - df[device2]
    
    mean_diff = df['difference'].mean()
    std_diff = df['difference'].std()
    
    activity_colors = {'Resting': 'blue', 'Exercise': '#C11C84', 'Recovery': 'orange'}
    
    plt.figure(figsize=(10, 8))
    
    if not full_data:
        for activity_type, color in activity_colors.items():
            subset = df[df['Activity'] == activity_type]
            plt.scatter(subset['mean'], subset['difference'], c=color, alpha=0.6, edgecolors='w', linewidth=0.5, label=activity_type)
    else:
        plt.scatter(df['mean'], df['difference'], alpha=0.6, edgecolors='w', linewidth=0.5)
    
    plt.axhline(mean_diff, color='gray', linestyle='--')
    
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    plt.axhline(loa_upper, color='red', linestyle='--')
    plt.axhline(loa_lower, color='red', linestyle='--')

    plot_xlim = plt.xlim()
    xOutPlot = plot_xlim[1] + 0.05 * (plot_xlim[1] - plot_xlim[0])  # Adjust 0.05 as needed for positioning

    plt.text(xOutPlot, mean_diff - 1.96*std_diff, r'-'+str(1.96)+'SD:' + "\n" + "%.2f" % loa_lower, ha = "center", va = "center", fontweight='bold')
    plt.text(xOutPlot, mean_diff + 1.96*std_diff, r'+'+str(1.96)+'SD:' + "\n" + "%.2f" % loa_upper, ha = "center", va = "center", fontweight='bold')
    plt.text(xOutPlot, mean_diff, r'Mean:' + "\n" + "%.2f" % mean_diff, ha = "center", va = "center", fontweight='bold')
    
    plt.xlabel('Mean of Two Devices (bpm)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Difference {get_device_name(device1)} and {get_device_name(device2)}', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim([-60, 100])
    plt.legend(loc='upper left')
    
    plt.show()  

def make_df_bland_altman_subject_per_device(df, device_name):

    device_data = df[df['value.deviceName'] == device_name]
    return device_data[device_data['Activity'] != 'Unknown'][['value.heartRate', 'TimeInSeconds', 'key.userId']]


def data_for_bland_altman_subject(df, subject_df, device1_name, device2_name, percentage=False):
    
    device1_HR = make_df_bland_altman_subject_per_device(df, device1_name)
    device2_HR = make_df_bland_altman_subject_per_device(df, device2_name)

    if isinstance(device1_HR, pd.DataFrame) and isinstance(device2_HR, pd.DataFrame):
        merged_HR = pd.concat([device1_HR[['value.heartRate']].reset_index(drop=True), device2_HR[['value.heartRate', 'key.userId']].reset_index(drop=True)], axis=1)
        merged_HR.columns = [device1_name, device2_name, 'key.userId']
        return merged_HR
    else:
        raise TypeError("Both device1_HR and device2_HR must be DataFrames")
    

def make_bland_altman_plot(df, subject_df, device1_name, device2_name, percentage=False):
    
    filtered_df_device1_device2 = filter_on_activity(df, subject_df)
    filtered_df_device1_device2 = filter_common_timepoints(filtered_df_device1_device2, device1_name, device2_name)
    
    if not check_measurements_per_device_pair(filtered_df_device1_device2, device1_name, device2_name):
        print('Something goes wrong as number of measurements between devices does not match')

    device1_HR, device1_HR_resting, device1_HR_exercise, device1_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device1_name)
    device2_HR, device2_HR_resting, device2_HR_exercise, device2_HR_recovery = split_HR_data_per_activity(filtered_df_device1_device2, device2_name)

    pyCompare.blandAltman(device1_HR, device2_HR, percentage=percentage, title=str('Full ' + device1_name + ' vs ' + device2_name))
    pyCompare.blandAltman(device1_HR_resting, device2_HR_resting, percentage=percentage, title=str('Resting ' + device1_name + ' vs ' + device2_name))
    pyCompare.blandAltman(device1_HR_exercise, device2_HR_exercise, percentage=percentage, title=str('Exercise ' + device1_name + ' vs ' + device2_name))
    pyCompare.blandAltman(device1_HR_recovery, device2_HR_recovery, percentage=percentage, title=str('Recovery ' + device1_name + ' vs ' + device2_name))


def bland_altman_plot_subject(df, device1, device2, full_data = False):

    df['mean'] = df[[device1, device2]].mean(axis=1)
    df['difference'] = df[device1] - df[device2]
    
    mean_diff = df['difference'].mean()
    std_diff = df['difference'].std()
    
    subject_colors = {'5f672ff8-5950-48a8-8b5b-b385a0add8f2': '#1f77b4', 
                    '784db1af-fdef-4b1f-8c36-e339b667e6e8': '#ff7f0e', 
                    '7b24b3e2-86a3-4857-950f-159697ba8923': '#2ca02c',
                    '9f2e03a2-0c44-41ea-b93a-e1de91a2a4a0': '#d62728', 
                    '2c31dbb8-5985-480d-9a73-b7fcb922a15d': '#9467bd',
                    '8d4804ac-eeaa-489a-8033-9a2d4c07ec86': '#8c564b', 
                    '2b54b23c-11da-45c4-ba6a-9bde6ffcc182': '#e377c2',
                    'a37fedae-ba8f-4be9-a686-b9240f3166d7': '#7f7f7f', 
                    '81252cbd-852a-4448-acd0-287ffa1de226': '#bcbd22',                  
                    '309eb2f0-bb5e-4072-881e-8953444cbd1b': '#17becf', 
                    'a59e17c0-2fd3-4e19-9135-9e0afb740047': '#aec7e8',
                    '7fd42a22-81b6-4f33-95e6-81eacd8ac400': '#ffbb78', 
                    'fd4731fd-b390-467c-977b-48ca733f7d0a': '#98df8a'
                    }

    plt.figure(figsize=(10, 8))
    
    if not full_data:
        for subject, color in subject_colors.items():
            subset = df[df['key.userId'] == subject]
            plt.scatter(subset['mean'], subset['difference'], c=color, alpha=0.6, edgecolors='w', linewidth=0.5, label=subjects[subject])
    else:
        plt.scatter(df['mean'], df['difference'], alpha=0.6, edgecolors='w', linewidth=0.5)

    plt.axhline(mean_diff, color='gray', linestyle='--')
    
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    plt.axhline(loa_upper, color='red', linestyle='--')
    plt.axhline(loa_lower, color='red', linestyle='--')

    plot_xlim = plt.xlim()
    xOutPlot = plot_xlim[1] + 0.05 * (plot_xlim[1] - plot_xlim[0])  # Adjust 0.05 as needed for positioning

    plt.text(xOutPlot, mean_diff - 1.96*std_diff, r'-'+str(1.96)+'SD:' + "\n" + "%.2f" % loa_lower, ha = "center", va = "center", fontweight='bold')
    plt.text(xOutPlot, mean_diff + 1.96*std_diff, r'+'+str(1.96)+'SD:' + "\n" + "%.2f" % loa_upper, ha = "center", va = "center", fontweight='bold')
    plt.text(xOutPlot, mean_diff, r'Mean:' + "\n" + "%.2f" % mean_diff, ha = "center", va = "center", fontweight='bold')
    
    # plt.title(f'Bland-Altman Plot {get_device_name(device1)} vs. {get_device_name(device2)}')
    plt.xlabel('Mean of Two Devices (bpm)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Difference {get_device_name(device1)} and {get_device_name(device2)}', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim([-60, 100])

    # axes = plt.axes()
    # axes.set_ylim([-60, 100])
    # Add a legend
    # plt.legend(loc='upper left')
    
    plt.show()
