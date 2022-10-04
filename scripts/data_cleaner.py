import argparse
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from tkinter import Tk, filedialog


def initial_cleaning(input_df):
    # sort the values since for some reason observations sometimes mixed in time
    input_df['DateTime'] = input_df['DateTime'].astype(float)

    # some datetime manipulations
    input_df['Timestamp'] = input_df['DateTime'].apply(lambda x: datetime.timestamp(from_ordinal(x)))
    input_df['DateTime'] = input_df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))

    input_df.sort_values(by='Timestamp', inplace=True)
    input_df.reset_index(drop=True, inplace=True)

    return input_df


def from_ordinal(ordinal, _epoch=datetime(1899, 12, 30)):
    """Converts serial date-time to DateTime object.

    Parameters
    ----------
    ordinal : float or int
        Original serial date-time.
    _epoch : datetime
        Start of the count.
        NOTE: for some reason timestamp is shifted by 2 days
        backwards from 01-01-1900, that is why default value
        is set to 30-12-1899.
    """
    return _epoch + timedelta(days=ordinal)


def fcsrtt_data_cleaner(input_file_path, input_encoding="utf_16", input_sep=";"):
    """Performs data manipulation from the raw csv file. Transform data in a way
    that 1 row represents the single trial.

    Parameters
    ----------
    input_file_path : str
        Path to the csv file with the raw data.
    input_encoding : str
        Encoding of an input file.
    input_sep : str
        Delimiter to use for an input file.

    Returns
    ----------
    final_output : DataFrame
        Resulted DataFrame object.
    """

    input_df = pd.DataFrame({})
    for fpath in input_file_path:
        try:
            df = pd.read_csv(fpath, encoding=input_encoding, sep=input_sep)
            len(df['DateTime']) # check whether the `sep` was chosen right

            # drop first rows from the data with the technical info
            df = df[~df['DateTime'].astype(str).apply(lambda x: x.startswith('#'))]
            # remove extra columns with additional information
            df = df.loc[:, :"MsgValue3"]
            # df['fname'] = fpath.split('/')[-1]
            input_df = input_df.append(df)
        except:
            # exit the function if the input file cannot be opened
            print("\nError while reading the input file. Change the `encoding` or `sep` parameters in the script.")
            return None

    # # resulted data is encoded to 'utf_16', change if different
    # try:
    #     input_df = pd.read_csv(input_file_path, encoding=input_encoding, sep=input_sep)
    #     len(input_df['DateTime']) # check whether the `sep` was chosen right
    # except:
    #     # exit the function if the input file cannot be opened
    #     print("\nError while reading the input file. Change the `encoding` or `sep` parameters in the script.")
    #     return None

    # drop first rows from the data with the technical info
    # input_df = input_df[~input_df['DateTime'].astype(str).apply(lambda x: x.startswith('#'))]

    input_df = initial_cleaning(input_df)

    ids = input_df['IdLabel'][~input_df['IdLabel'].isnull()].unique()
    ids.sort()
    ids_dict = dict(input_df[['IdLabel', 'IdRFID']].drop_duplicates().dropna().values)

    final_output = pd.DataFrame({})

    for animal_id in tqdm(ids):
        # print(animal_id)
        indices_start = input_df[(input_df['IdLabel'] == animal_id) & (input_df['SystemMsg'] == 'start exp')].index
        indices_end = input_df[(input_df['IdLabel'] == animal_id) & (input_df['SystemMsg'] == 'end exp')].index

        for session_i in range(len(indices_start)):

            try:
                ind_start = indices_start[session_i]
                ind_end = indices_end[session_i]
                subj_data = input_df.iloc[ind_start:ind_end+1, :].reset_index(drop=True)
            except:
                continue

            # print(animal_id)
            # time.sleep(0.5)
            # print(f'Animals: {animal_id}, session: {session_i}')

            if 'Unexpected' in subj_data['SystemMsg'].values:
                # print('all good')
                continue

            if 'start trial 1' not in subj_data['SystemMsg'].values:
                # print('all good')
                continue

            cndtn = subj_data['SystemMsg'].apply(
                lambda x: x.startswith('start trial') if type(x) == str else False)
            total_trials = subj_data['SystemMsg'][cndtn].apply(lambda x: int(x.split(' ')[2])).max()
            total_outcomes = (subj_data['SystemMsg'] == 'Reward?').sum()

            wait_poke_ts = subj_data['Timestamp'][subj_data['SystemMsg'] == 'wait poke'] \
                .reset_index(drop=True)

            trial_start_ts = subj_data['Timestamp'][cndtn] \
                .reset_index(drop=True)
            trial_start_ts.name = 'trialStart'

            trial_end_ts = subj_data['Timestamp'][subj_data['SystemMsg'] == 'start iti'] \
                .reset_index(drop=True)
            if len(trial_start_ts) != len(trial_end_ts):
                trial_end_ts = trial_end_ts.append(
                    pd.Series(
                        subj_data['Timestamp'][subj_data['SystemMsg'] == 'end exp']
                    ),
                    ignore_index=True)
            trial_end_ts.name = 'trialEnd'

            if len(wait_poke_ts) != len(trial_start_ts):
                wait_poke_ts = wait_poke_ts[:len(wait_poke_ts)-1]

            start_latency = trial_start_ts - wait_poke_ts
            start_latency.name = 'startLatency'

            trial_duration = trial_end_ts - trial_start_ts
            trial_duration.name = 'trialDuration'

            decision = subj_data['MsgValue1'][subj_data['SystemMsg'] == 'decision:'] \
                .reset_index(drop=True)
            decision_n = decision.apply(lambda x: x.split(' ')[1])
            decision_n.name = 'decisionNumber'
            decision_pos = decision.apply(lambda x: x.split(' ')[2][2])
            decision_pos.name = 'decisionPosition'
            decision_img = decision.apply(lambda x: x.split(' ')[2][4])
            decision_img.name = 'decisionImage'

            decision_ts = subj_data['Timestamp'][subj_data['SystemMsg'] == 'decision:'] \
                .reset_index(drop=True)
            decision_latency = decision_ts - trial_start_ts
            decision_latency.name = 'decisionLatency'

            reward = subj_data['MsgValue1'][subj_data['SystemMsg'] == 'Reward?'] \
                .reset_index(drop=True)
            reward.name = 'reward'
            reward = reward == 'True'

            reward_ready_ts = subj_data['Timestamp'][subj_data['SystemMsg'] == 'reward ready'] \
                .reset_index(drop=True)

            reward_collected_ts = subj_data['Timestamp'][subj_data['SystemMsg'] == 'reward collected'] \
                .reset_index(drop=True)

            reward_latency = reward_collected_ts - reward_ready_ts
            reward_latency.index = reward[reward == True].index
            reward_latency.name = 'rewardLatency'

            opt1 = subj_data['MsgValue1'][cndtn]\
                .reset_index(drop=True)\
                .apply(lambda x: x.split(' ')[2])
            opt1.name = 'Option1'

            opt2 = subj_data['MsgValue2'][cndtn]\
                .reset_index(drop=True)\
                .apply(lambda x: x.split(' ')[2])
            opt2.name = 'Option2'

            # opt3 = subj_data['MsgValue3'][cndtn]\
            #     .reset_index(drop=True)\
            #     .apply(lambda x: x.split(' ')[2])
            # opt3.name = 'Option3'

            p1 = subj_data['MsgValue1'][cndtn] \
                .reset_index(drop=True) \
                .apply(lambda x: x.split('=')[1])
            p1.name = 'P1'

            p2 = subj_data['MsgValue2'][cndtn] \
                .reset_index(drop=True) \
                .apply(lambda x: x.split('=')[1])
            p2.name = 'P2'

            # p3 = subj_data['MsgValue3'][cndtn] \
            #     .reset_index(drop=True) \
            #     .apply(lambda x: x.split('=')[1])
            # p3.name = 'P3'

            correction_trial = subj_data['MsgValue1'][subj_data['SystemMsg'] == 'Correction trial ']\
                .reset_index(drop=True)
            correction_trial.name = 'correctionTrial'

            session_out = pd.concat(
                [
                    trial_start_ts, trial_end_ts, trial_duration, start_latency, opt1, opt2,
                    # opt3,p3,
                    p1, p2, decision_n, decision_pos, decision_img, decision_latency, reward,
                    correction_trial
                ],
                axis=1)
            session_out = session_out.join(reward_latency)

            if total_trials != total_outcomes:
                session_out = session_out.iloc[:total_trials-1, :]

            session_out['trial'] = session_out.index + 1
            session_out['animalID'] = animal_id
            session_out['session'] = session_i + 1
            session_out['scenario'] = subj_data['MsgValue2'][0]

            final_output = final_output.append(session_out).reset_index(drop=True)

    return final_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments for specification of input file details.')
    parser.add_argument(
        '-s', '--sep',
        dest="separator", type=str,
        default=';',
        help='Separator in the file(s) (, ; \\t)'
    )
    parser.add_argument(
        '-e', '--encoding',
        dest="encoding", type=str,
        default='utf_16',
        help='Encoding of the input file(s) (utf_16, utf_8)'
    )

    parser.add_argument(
        '-o', '--output',
        dest="output", type=str,
        default=' ',
        help='Output file name'
    )

    args = parser.parse_args()

    # interactive selection of an input file and output folder
    # output file will be saved with the current time in a name
    root = Tk()
    root.withdraw()
    input_files = filedialog.askopenfilenames(title='Choose the input file(s)')
    output_path = filedialog.askdirectory(title='Choose the directory for the output file')

    if args.output == ' ':
        output_name = input_files[0].split('/')[-1].replace(' ', '_').replace('.csv', '_PROCESSED.csv')
    else:
        output_name = args.output

    output_file = f"{output_path}/{output_name}"

    print("\n" + "="*40)
    print(f"Input file(s): {input_files}")
    print(f"Output folder: {output_path}")
    print("="*40)

    final_output = fcsrtt_data_cleaner(
        input_file_path=input_files,
        input_encoding=args.encoding,
        input_sep=args.separator)

    if final_output is not None:
        final_output = final_output[[
            'animalID', 'session', 'scenario', 'trial', 'correctionTrial',
            'trialStart', 'trialEnd', 'trialDuration',
            'startLatency', 'Option1', 'Option2',
            # 'Option3',
            'P1', 'P2',
            # 'P3',
            'decisionNumber', 'decisionPosition', 'decisionImage',
            'decisionLatency', 'reward', 'rewardLatency'
        ]].round(2)

        # final_output['trialStart'] = pd.to_datetime(final_output['trialStart'], unit='s')
        # final_output['trialEnd'] = pd.to_datetime(final_output['trialEnd'], unit='s')

        final_output['trialStart'] = final_output['trialStart'].apply(lambda x: datetime.fromtimestamp(x))
        final_output['trialEnd'] = final_output['trialEnd'].apply(lambda x: datetime.fromtimestamp(x))

        final_output.to_csv(output_file, index=False)
        print("\nOutput file was saved successfully!")
        print(f"File path: {output_file}")
