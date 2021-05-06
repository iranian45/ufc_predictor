# import packages
import datetime
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import timedelta

# set max columns to none to review layout
pd.set_option('display.max_columns', None, 'display.max_rows', None)

# Define weight classes and create def
weight_classes = ['Atomweight', 'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight',
                  'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight', 'Open Weight', 'Catch Weight']


def extract(x):
    for item in weight_classes:
        if item in x:
            return item


# Import csv files
df_fight_info = pd.read_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                            '\\fight_info.csv')
df_fighter_info = pd.read_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                              '\\fighter_details.csv')
df_fight_details = pd.read_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                               '\\fight_details.csv')
df_next_fight_details = pd.read_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                                    '\\next_fight_info.csv')
# Filter Next Fight Info for upcoming fights
df_next_fight_details['Fight_Date'] = pd.to_datetime(df_next_fight_details.Fight_Date)
today = datetime.datetime.today()
next_week = today + timedelta(weeks=1)
df_next_fight_details = df_next_fight_details[(df_next_fight_details['Fight_Date'] >= datetime.datetime.today())]
df_next_fight_details = df_next_fight_details[(df_next_fight_details['Fight_Date'] <= next_week)]

# Merge dfs
df_perf = pd.merge(
    df_fight_details,
    df_fight_info,
    on='Fight_URL',
    how='inner'
)

df_perf = df_perf.append(df_next_fight_details, ignore_index=True)

df_perf = pd.merge(
    df_perf,
    df_fighter_info,
    how='left',
    left_on=['F1_URL'],
    right_on=['Fighter_URL']
)

# remove duplicate columns
df_perf.drop(['Fighter_URL'], axis=1, inplace=True)
# Rename columns for F1
df_perf.rename(columns={'Height': 'F1_Height', 'Weight': 'F1_Weight',
                        'Reach': 'F1_Reach', 'Stance': 'F1_Stance', 'DoB': 'F1_DoB'}, inplace=True)
df_perf = pd.merge(
    df_perf,
    df_fighter_info,
    how='left',
    left_on=['F2_URL'],
    right_on=['Fighter_URL']
)
# remove duplicate columns
df_perf.drop(['Fighter_URL'], axis=1, inplace=True)

# Rename columns for F2
df_perf.rename(columns={'Height': 'F2_Height', 'Weight': 'F2_Weight',
                        'Reach': 'F2_Reach', 'Stance': 'F2_Stance', 'DoB': 'F2_DoB'}, inplace=True)

# Reorganize Data
df_perf = df_perf[['Fight_Title', 'Fight_URL', 'Fight_Date', 'Event_Name', 'Event_URL', 'Method', 'Round_Finished',
                   'Referee', 'Stop_Time', 'F1_URL', 'F1_First', 'F1_Last', 'F1_Height', 'F1_Weight', 'F1_Reach',
                   'F1_Stance', 'F1_DoB', 'F1_Status', 'F1_KD', 'F1_Total_Str_Landed', 'F1_Total_Str_Attempted',
                   'F1_Sig_Str_Landed', 'F1_Sig_Str_Attempted', 'F1_Sig_Str_Head_Landed', 'F1_Sig_Str_Head_Attempted',
                   'F1_Sig_Str_Body_Landed', 'F1_Sig_Str_Body_Attempted', 'F1_Sig_Str_Leg_Landed',
                   'F1_Sig_Str_Leg_Attempted', 'F1_Sig_Str_Distance_Landed', 'F1_Sig_Str_Distance_Attempted',
                   'F1_Sig_Str_Clinch_Landed', 'F1_Sig_Str_Clinch_Attempted', 'F1_Sig_Str_Ground_Landed',
                   'F1_Sig_Str_Ground_Attempted', 'F1_TD_Landed', 'F1_TD_Attempted', 'F1_Sub_Attempted', 'F1_Rev',
                   'F1_CTRL_TIME', 'F2_URL', 'F2_First', 'F2_Last', 'F2_Height', 'F2_Weight', 'F2_Reach', 'F2_Stance',
                   'F2_DoB', 'F2_Status', 'F2_KD', 'F2_Total_Str_Landed', 'F2_Total_Str_Attempted', 'F2_Sig_Str_Landed',
                   'F2_Sig_Str_Attempted', 'F2_Sig_Str_Head_Landed', 'F2_Sig_Str_Head_Attempted',
                   'F2_Sig_Str_Body_Landed', 'F2_Sig_Str_Body_Attempted', 'F2_Sig_Str_Leg_Landed',
                   'F2_Sig_Str_Leg_Attempted', 'F2_Sig_Str_Distance_Landed', 'F2_Sig_Str_Distance_Attempted',
                   'F2_Sig_Str_Clinch_Landed', 'F2_Sig_Str_Clinch_Attempted', 'F2_Sig_Str_Ground_Landed',
                   'F2_Sig_Str_Ground_Attempted', 'F2_TD_Landed', 'F2_TD_Attempted', 'F2_Sub_Attempted', 'F2_Rev',
                   'F2_CTRL_TIME']]

# Replace bad data to None and Convert Types
# df_perf = df_perf.replace({np.nan: None})
df_perf.replace('--', None, inplace=True)
df_perf['Fight_Title'] = df_perf['Fight_Title'].astype(str)
df_perf['Fight_Date'] = df_perf['Fight_Date'].astype('datetime64[ns]')
df_perf['Stop_Time'] = pd.to_datetime(df_perf['Stop_Time'], format='%M:%S') - \
                       pd.to_datetime(df_perf['Stop_Time'], format='%M:%S').dt.normalize()
df_perf['F1_Height'] = (pd.to_numeric(df_perf['F1_Height'].str.split("'").str[0]) * 12) \
                       + pd.to_numeric(df_perf['F1_Height'].str.split("'").str[1].str.split('"').str[0])
df_perf['F1_Weight'] = pd.to_numeric(df_perf['F1_Weight'])
df_perf['F1_Reach'] = pd.to_numeric(df_perf['F1_Reach'].str.strip('"'))
df_perf['F1_DoB'] = df_perf['F1_DoB'].astype('datetime64[ns]')
df_perf['F1_CTRL_TIME'] = (pd.to_datetime(df_perf['F1_CTRL_TIME'], format='%M:%S') -
                           pd.to_datetime(df_perf['F1_CTRL_TIME'], format='%M:%S').dt.normalize()) / np.timedelta64(1,
                                                                                                                    's')
df_perf['F2_Height'] = (pd.to_numeric(df_perf['F2_Height'].str.split("'").str[0]) * 12) \
                       + pd.to_numeric(df_perf['F2_Height'].str.split("'").str[1].str.split('"').str[0])
df_perf['F2_Weight'] = pd.to_numeric(df_perf['F2_Weight'])
df_perf['F2_Reach'] = pd.to_numeric(df_perf['F2_Reach'].str.strip('"'))
df_perf['F2_DoB'] = df_perf['F2_DoB'].astype('datetime64[ns]')
df_perf['F2_CTRL_TIME'] = (pd.to_datetime(df_perf['F2_CTRL_TIME'], format='%M:%S') -
                           pd.to_datetime(df_perf['F2_CTRL_TIME'], format='%M:%S').dt.normalize()) / np.timedelta64(1,
                                                                                                                    's')

# duplicate rows so each fighter is in F1 for all fights
f1_cols = list(filter(re.compile(r'F1_').search, df_perf.columns))
f2_cols = list(filter(re.compile(r'F2_').search, df_perf.columns))
non_f_cols = ['Fight_Title', 'Fight_URL', 'Fight_Date', 'Event_Name', 'Event_URL', 'Method', 'Round_Finished',
              'Referee', 'Stop_Time']
inverse_df = df_perf[[*non_f_cols, *f2_cols, *f1_cols]]
inverse_df.columns = df_perf.columns
df = pd.concat((df_perf, inverse_df)).sort_index().reset_index(drop=True)

df.drop_duplicates(inplace=True, ignore_index=True)
# df.query('F1_URL =="http://www.ufcstats.com/fighter-details/1338e2c7480bdf9e"', inplace=True)

# Add calculated columns
df['F1_Age'] = (df.Fight_Date - df.F1_DoB).astype('m8[Y]')
df['F2_Age'] = (df.Fight_Date - df.F2_DoB).astype('m8[Y]')
# df['Fight_Duration'] = ((df['Round_Finished'] - 1) * pd.Timedelta(minutes=5) + df['Stop_Time']) / np.timedelta64(1, 's')
df['Weight_Class'] = df['Fight_Title'].apply(lambda x: extract(x))


def f1_sub_landed(row):
    if row['Method'] == ' Submission ':
        if row['F1_Status'] == 'W':
            return 1
    else:
        return 0


df['F1_Sub_Landed'] = df.apply(lambda row: f1_sub_landed(row), axis=1)


def f2_sub_landed(row):
    if row['Method'] == 'Submission':
        if row['F2_Status'] == 'W':
            return 1
    else:
        return 0


df['F2_Sub_Landed'] = df.apply(lambda row: f2_sub_landed(row), axis=1)

# # Filter Data
df.query('Fight_Date >="2010-01-01"', inplace=True)

# Calculate Deltas
df_Minimized = df.sort_values(by='Fight_Date')
df_Minimized['Height_Delta'] = (df_Minimized.F1_Height - df_Minimized.F2_Height)
df_Minimized['Weight_Delta'] = (df_Minimized.F1_Weight - df_Minimized.F2_Weight)
df_Minimized['Reach_Delta'] = (df_Minimized.F1_Reach - df_Minimized.F2_Reach)
df_Minimized['Age_Delta'] = (df_Minimized.F1_Age - df_Minimized.F2_Age)

# Cumulative Average Stats prior to current fight
df_Minimized['F1_Fight_Count'] = df_Minimized.groupby(['F1_URL']).cumcount()
df_Minimized['F2_Fight_Count'] = df_Minimized.groupby(['F2_URL']).cumcount()

df_Minimized['F1_KD_Cum'] = df_Minimized[{'F1_URL', 'F1_KD'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_KD'] = (df_Minimized['F1_KD_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_KD_Cum'] = df_Minimized[{'F2_URL', 'F2_KD'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_KD'] = (df_Minimized['F2_KD_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)

df_Minimized['F1_Total_Str_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Total_Str_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Str_Landed'] = (df_Minimized['F1_Total_Str_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F2_Total_Str_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Total_Str_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Str_Landed'] = (df_Minimized['F2_Total_Str_Landed_Cum'] / df_Minimized['F2_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F1_Total_Str_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Total_Str_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Str_Attempted'] = (
        df_Minimized['F1_Total_Str_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Total_Str_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Total_Str_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Str_Attempted'] = (
        df_Minimized['F2_Total_Str_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Str_Accuracy'] = (
        df_Minimized['F1_Total_Str_Landed_Cum'] / df_Minimized['F1_Total_Str_Attempted_Cum']).fillna(0).round(2)
df_Minimized['F2_Avg_Str_Accuracy'] = (
        df_Minimized['F2_Total_Str_Landed_Cum'] / df_Minimized['F2_Total_Str_Attempted_Cum']).fillna(0).round(2)

df_Minimized['F1_Sig_Str_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Landed'] = (df_Minimized['F1_Sig_Str_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F2_Sig_Str_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Landed'] = (df_Minimized['F2_Sig_Str_Landed'] / df_Minimized['F2_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F1_Sig_Str_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Attempted'] = (
        df_Minimized['F1_Sig_Str_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Attempted'] = (
        df_Minimized['F2_Sig_Str_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Landed_Cum'] / df_Minimized['F1_Sig_Str_Attempted_Cum']).fillna(0).round(2)
df_Minimized['F2_Avg_Sig_Str_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Landed_Cum'] / df_Minimized['F2_Sig_Str_Attempted_Cum']).fillna(0).round(2)

df_Minimized['F1_Sig_Str_Head_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Head_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Head_Landed'] = (
        df_Minimized['F1_Sig_Str_Head_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Head_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Head_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Head_Landed'] = (
        df_Minimized['F2_Sig_Str_Head_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Head_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Head_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Head_Attempted'] = (
        df_Minimized['F1_Sig_Str_Head_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Head_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Head_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Head_Attempted'] = (
        df_Minimized['F2_Sig_Str_Head_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Head_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Head_Landed_Cum'] / df_Minimized['F1_Sig_Str_Head_Attempted_Cum']).fillna(0).round(
    2)
df_Minimized['F2_Avg_Sig_Str_Head_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Head_Landed_Cum'] / df_Minimized['F2_Sig_Str_Head_Attempted_Cum']).fillna(0).round(
    2)

df_Minimized['F1_Sig_Str_Body_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Body_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Body_Landed'] = (
        df_Minimized['F1_Sig_Str_Body_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Body_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Body_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Body_Landed'] = (
        df_Minimized['F2_Sig_Str_Body_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Body_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Body_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Body_Attempted'] = (
        df_Minimized['F1_Sig_Str_Body_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Body_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Body_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Body_Attempted'] = (
        df_Minimized['F2_Sig_Str_Body_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Body_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Body_Landed_Cum'] / df_Minimized['F1_Sig_Str_Body_Attempted_Cum']).fillna(0).round(
    2)
df_Minimized['F2_Avg_Sig_Str_Body_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Body_Landed_Cum'] / df_Minimized['F2_Sig_Str_Body_Attempted_Cum']).fillna(0).round(
    2)

df_Minimized['F1_Sig_Str_Leg_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Leg_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Leg_Landed'] = (
        df_Minimized['F1_Sig_Str_Leg_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Leg_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Leg_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Leg_Landed'] = (
        df_Minimized['F2_Sig_Str_Leg_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Leg_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Leg_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Leg_Attempted'] = (
        df_Minimized['F1_Sig_Str_Leg_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Leg_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Leg_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Leg_Attempted'] = (
        df_Minimized['F2_Sig_Str_Leg_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Leg_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Leg_Landed_Cum'] / df_Minimized['F1_Sig_Str_Leg_Attempted_Cum']).fillna(0).round(2)
df_Minimized['F2_Avg_Sig_Str_Leg_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Leg_Landed_Cum'] / df_Minimized['F2_Sig_Str_Leg_Attempted_Cum']).fillna(0).round(2)

df_Minimized['F1_Sig_Str_Distance_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Distance_Landed'}].groupby(
    'F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Distance_Landed'] = (
        df_Minimized['F1_Sig_Str_Distance_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Distance_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Distance_Landed'}].groupby(
    'F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Distance_Landed'] = (
        df_Minimized['F2_Sig_Str_Distance_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Distance_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Distance_Attempted'}].groupby(
    'F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Distance_Attempted'] = (
        df_Minimized['F1_Sig_Str_Distance_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Distance_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Distance_Attempted'}].groupby(
    'F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Distance_Attempted'] = (
        df_Minimized['F2_Sig_Str_Distance_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Distance_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Distance_Landed_Cum'] / df_Minimized['F1_Sig_Str_Distance_Attempted_Cum']).fillna(
    0).round(2)
df_Minimized['F2_Avg_Sig_Str_Distance_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Distance_Landed_Cum'] / df_Minimized['F2_Sig_Str_Distance_Attempted_Cum']).fillna(
    0).round(2)

df_Minimized['F1_Sig_Str_Clinch_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Clinch_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Clinch_Landed'] = (
        df_Minimized['F1_Sig_Str_Clinch_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Clinch_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Clinch_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Clinch_Landed'] = (
        df_Minimized['F2_Sig_Str_Clinch_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Clinch_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Clinch_Attempted'}].groupby(
    'F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Clinch_Attempted'] = (
        df_Minimized['F1_Sig_Str_Clinch_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Clinch_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Clinch_Attempted'}].groupby(
    'F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Clinch_Attempted'] = (
        df_Minimized['F2_Sig_Str_Clinch_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Clinch_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Clinch_Landed_Cum'] / df_Minimized['F1_Sig_Str_Clinch_Attempted_Cum']).fillna(
    0).round(2)
df_Minimized['F2_Avg_Sig_Str_Clinch_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Clinch_Landed_Cum'] / df_Minimized['F2_Sig_Str_Clinch_Attempted_Cum']).fillna(
    0).round(2)

df_Minimized['F1_Sig_Str_Ground_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Ground_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Ground_Landed'] = (
        df_Minimized['F1_Sig_Str_Ground_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Ground_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Ground_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Ground_Landed'] = (
        df_Minimized['F2_Sig_Str_Ground_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sig_Str_Ground_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sig_Str_Ground_Attempted'}].groupby(
    'F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sig_Str_Ground_Attempted'] = (
        df_Minimized['F1_Sig_Str_Ground_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Sig_Str_Ground_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sig_Str_Ground_Attempted'}].groupby(
    'F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sig_Str_Ground_Attempted'] = (
        df_Minimized['F2_Sig_Str_Ground_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Avg_Sig_Str_Ground_Accuracy'] = (
        df_Minimized['F1_Sig_Str_Ground_Landed_Cum'] / df_Minimized['F1_Sig_Str_Ground_Attempted_Cum']).fillna(
    0).round(2)
df_Minimized['F2_Avg_Sig_Str_Ground_Accuracy'] = (
        df_Minimized['F2_Sig_Str_Ground_Landed_Cum'] / df_Minimized['F2_Sig_Str_Ground_Attempted_Cum']).fillna(
    0).round(2)

df_Minimized['F1_TD_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_TD_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_TD_Landed'] = (df_Minimized['F1_TD_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(
    2)
df_Minimized['F2_TD_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_TD_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_TD_Landed'] = (df_Minimized['F2_TD_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_TD_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_TD_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_TD_Attempted'] = (df_Minimized['F1_TD_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F2_TD_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_TD_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_TD_Attempted'] = (df_Minimized['F2_TD_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F1_Avg_TD_Accuracy'] = (df_Minimized['F1_TD_Landed_Cum'] / df_Minimized['F1_TD_Attempted_Cum']).fillna(
    0).round(2)
df_Minimized['F2_Avg_TD_Accuracy'] = (df_Minimized['F2_TD_Landed_Cum'] / df_Minimized['F2_TD_Attempted_Cum']).fillna(
    0).round(2)

df_Minimized['F1_Sub_Landed_Cum'] = df_Minimized[{'F1_URL', 'F1_Sub_Landed'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sub_Landed'] = (df_Minimized['F1_Sub_Landed_Cum'] / df_Minimized['F1_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F2_Sub_Landed_Cum'] = df_Minimized[{'F2_URL', 'F2_Sub_Landed'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sub_Landed'] = (df_Minimized['F2_Sub_Landed'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)
df_Minimized['F1_Sub_Attempted_Cum'] = df_Minimized[{'F1_URL', 'F1_Sub_Attempted'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Sub_Attempted'] = (df_Minimized['F1_Sub_Attempted_Cum'] / df_Minimized['F1_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F2_Sub_Attempted_Cum'] = df_Minimized[{'F2_URL', 'F2_Sub_Attempted'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Sub_Attempted'] = (df_Minimized['F2_Sub_Attempted_Cum'] / df_Minimized['F2_Fight_Count']).fillna(
    0).round(2)
df_Minimized['F1_Avg_Sub_Accuracy'] = (df_Minimized['F1_Sub_Landed_Cum'] / df_Minimized['F1_Sub_Attempted_Cum']).fillna(
    0).round(2)
df_Minimized['F2_Avg_Sub_Accuracy'] = (df_Minimized['F2_Sub_Landed_Cum'] / df_Minimized['F2_Sub_Attempted_Cum']).fillna(
    0).round(2)

df_Minimized['F1_Rev_Cum'] = df_Minimized[{'F1_URL', 'F1_Rev'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_Rev'] = (df_Minimized['F1_Rev_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(2)
df_Minimized['F2_Rev_Cum'] = df_Minimized[{'F2_URL', 'F2_Rev'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_Rev'] = (df_Minimized['F2_Rev_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(2)

df_Minimized['F1_CTRL_TIME_Cum'] = df_Minimized[{'F1_URL', 'F1_CTRL_TIME'}].groupby('F1_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F1_avg_CTRL_TIME'] = (df_Minimized['F1_CTRL_TIME_Cum'] / df_Minimized['F1_Fight_Count']).fillna(0).round(
    2)
df_Minimized['F2_CTRL_TIME_Cum'] = df_Minimized[{'F2_URL', 'F2_CTRL_TIME'}].groupby('F2_URL'). \
    apply(lambda x: x.shift().cumsum()).fillna(0)
df_Minimized['F2_avg_CTRL_TIME'] = (df_Minimized['F2_CTRL_TIME_Cum'] / df_Minimized['F2_Fight_Count']).fillna(0).round(
    2)

df_Minimized = df_Minimized[
    ['F1_URL', 'F1_Status', 'F2_URL', 'F2_Status', 'F1_Age', 'F2_Age', 'Height_Delta', 'Weight_Delta', 'Reach_Delta',
     'Age_Delta', 'F1_avg_KD', 'F2_avg_KD', 'F1_avg_Str_Landed', 'F2_avg_Str_Landed',
     'F1_avg_Str_Attempted', 'F2_avg_Str_Attempted', 'F1_Avg_Str_Accuracy',
     'F2_Avg_Str_Accuracy', 'F1_avg_Sig_Str_Landed', 'F2_avg_Sig_Str_Landed',
     'F1_avg_Sig_Str_Attempted', 'F2_avg_Sig_Str_Attempted', 'F1_Avg_Sig_Str_Accuracy',
     'F2_Avg_Sig_Str_Accuracy', 'F1_avg_Sig_Str_Head_Landed', 'F2_avg_Sig_Str_Head_Landed',
     'F1_avg_Sig_Str_Head_Attempted', 'F2_avg_Sig_Str_Head_Attempted',
     'F1_Avg_Sig_Str_Head_Accuracy', 'F2_Avg_Sig_Str_Head_Accuracy',
     'F1_avg_Sig_Str_Body_Landed', 'F2_avg_Sig_Str_Body_Landed',
     'F1_avg_Sig_Str_Body_Attempted', 'F2_avg_Sig_Str_Body_Attempted',
     'F1_Avg_Sig_Str_Body_Accuracy', 'F2_Avg_Sig_Str_Body_Accuracy',
     'F1_avg_Sig_Str_Leg_Landed', 'F2_avg_Sig_Str_Leg_Landed', 'F1_avg_Sig_Str_Leg_Attempted',
     'F2_avg_Sig_Str_Leg_Attempted', 'F1_Avg_Sig_Str_Leg_Accuracy',
     'F2_Avg_Sig_Str_Leg_Accuracy', 'F1_avg_Sig_Str_Distance_Landed',
     'F2_avg_Sig_Str_Distance_Landed', 'F1_avg_Sig_Str_Distance_Attempted',
     'F2_avg_Sig_Str_Distance_Attempted', 'F1_Avg_Sig_Str_Distance_Accuracy',
     'F2_Avg_Sig_Str_Distance_Accuracy', 'F1_avg_Sig_Str_Clinch_Landed',
     'F2_avg_Sig_Str_Clinch_Landed', 'F1_avg_Sig_Str_Clinch_Attempted',
     'F2_avg_Sig_Str_Clinch_Attempted', 'F1_Avg_Sig_Str_Clinch_Accuracy',
     'F2_Avg_Sig_Str_Clinch_Accuracy', 'F1_avg_Sig_Str_Ground_Landed',
     'F2_avg_Sig_Str_Ground_Landed', 'F1_avg_Sig_Str_Ground_Attempted',
     'F2_avg_Sig_Str_Ground_Attempted', 'F1_Avg_Sig_Str_Ground_Accuracy',
     'F2_Avg_Sig_Str_Ground_Accuracy', 'F1_avg_TD_Landed', 'F2_avg_TD_Landed',
     'F1_avg_TD_Attempted', 'F2_avg_TD_Attempted', 'F1_Avg_TD_Accuracy', 'F2_Avg_TD_Accuracy',
     'F1_avg_Sub_Landed', 'F2_avg_Sub_Landed', 'F1_avg_Sub_Attempted', 'F2_avg_Sub_Attempted',
     'F1_Avg_Sub_Accuracy', 'F2_Avg_Sub_Accuracy', 'F1_avg_Rev', 'F2_avg_Rev',
     'F1_avg_CTRL_TIME', 'F2_avg_CTRL_TIME']]

df_Minimized.F1_Status = pd.Categorical(df_Minimized.F1_Status)
df_Minimized.F2_Status = pd.Categorical(df_Minimized.F2_Status)

# Copy DF for Prediction Calc
df_Pred = df_Minimized[df_Minimized.F1_Status.isnull()]
df_Minimized = df_Minimized[df_Minimized.F1_Status.notnull()]


df_Minimized.loc[df_Minimized.F1_Status == 'W', 'Winner'] = 1
df_Minimized.loc[df_Minimized.F2_Status == 'W', 'Winner'] = 2
final = df_Minimized.drop(df_Minimized[{'F1_Status', 'F2_Status'}], axis=1)
df_Pred = df_Pred.drop(df_Pred[{'F1_Status', 'F2_Status'}], axis=1)
# df_Pred['Winner'] = None

# df_Pred.to_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
#                             '\\test2.csv')

# # final = pd.get_dummies(final, prefix=['F1_URL', 'F2_URL'], columns=['F1_URL', 'F2_URL'])

final = final.drop(final[{'F1_URL', 'F2_URL'}], axis=1)

X = final.drop(['Winner'], axis=1)
X = X.notna()
y = final['Winner']
y = y.notna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set Accuracy: ", '%.3f' % score)
print("Test set accuracy", '%.3f' % score2)

pred_set = df_Pred
backup_pred_set = pred_set[{'F1_URL', 'F2_URL'}]
pred_set = pred_set.drop(pred_set[{'F1_URL', 'F2_URL'}], axis=1)

# missing_cols = set(final.columns) - set(pred_set.columns)
# for c in missing_cols:
#     pred_set[c] = 0
# pred_set = pred_set[final.columns]
#
# pred_set = pred_set.drop(['Winner'], axis=1)

# print(pred_set.columns)
predictions = logreg.predict(pred_set)
print(predictions)
pred_set.to_csv('F:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                            '\\predset.csv')
#
# print(backup_pred_set.iloc[10, 0], " 0 ")
# print(backup_pred_set.iloc[10, 1], " 1 ")
# # print(backup_pred_set.iloc[10, 2], " 2 ")
# for i in range(df_Pred.shape[0]):
#     print(backup_pred_set.iloc[i, 0], " and ", backup_pred_set.iloc[i, 1])
#     if predictions[i] == 1:
#         print("Winner: ", backup_pred_set.iloc[i, 0])
#
#     else:
#         print("Winner: ", backup_pred_set.iloc[i, 1])
#     print("")
