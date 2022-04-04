import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog

if __name__ == "__main__":

    root = Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(title='Choose the input file')
    plots_path = filedialog.askdirectory(title='Choose the directory for resulting plots')

    animal_data = pd.read_csv(input_file)

    animal_data = animal_data[animal_data['animalID'] != 'Test2']
    ids = animal_data['animalID'].unique()
    animal_data.sort_values(by=['animalID', 'session', 'trial'], inplace=True)

    animal_data.insert(3, 'trialTotal', np.NaN)
    animal_data['trialTotal']  = animal_data.groupby('animalID').cumcount()+1

    animal_data.insert(4, 'trialByScenario', np.NaN)
    animal_data['trialByScenario']  = animal_data.groupby(['animalID', 'scenario']).cumcount()+1

    max_trial = animal_data.trialTotal.max()

    animal_data['optimal_choice'] = animal_data['decisionNumber'] \
        .apply(lambda x: x == 1)


    # latencies plots
    for metric in ['decisionLatency', 'rewardLatency', 'startLatency']:

        plt.figure(figsize=(35,15))

        for (i, animal_id) in enumerate(ids):
            temp_df = animal_data[animal_data['animalID'] == animal_id]
            plt.subplot(len(ids), 1, i+1)
            plt.plot(
                temp_df['trialTotal'],
                temp_df[metric])
            for ses_start in temp_df['trialTotal'][temp_df['trial'] == 1]:
                plt.axvline(ses_start, color='black')
            plt.title(animal_id, fontweight='bold', fontsize=14)
            plt.xlim([0, max_trial])
            # plt.ylabel('Seconds')

        plt.xlabel('Trial')
        plt.suptitle(metric + ' [s]', fontweight='bold', fontsize=16)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plots_path}/{metric}ByTrials.png')
        plt.close()
    #%%
    # plobability of the first option plot

    plt.figure(figsize=(35,15))

    for (i, animal_id) in enumerate(ids):
        temp_df = animal_data[animal_data['animalID'] == animal_id] \
            .reset_index(drop=True)

        sessions = temp_df['session'].unique()
        prob_of_first = np.array([])

        for s in sessions:
            vals = (temp_df[temp_df['session'] == s]['decisionNumber'] == 1) \
                .expanding() \
                .mean() \
                .values
            prob_of_first = np.append(prob_of_first, vals)

        plt.subplot(len(ids), 1, i+1)
        plt.plot(
            temp_df['trialTotal'],
            prob_of_first)
        for ses_start in temp_df['trialTotal'][temp_df['trial'] == 1]:
            plt.axvline(ses_start, color='black')
        plt.ylim([-0.1,1.1])
        plt.title(animal_id, fontweight='bold', fontsize=14)
        plt.xlim([0, max_trial])

    plt.xlabel('Trial')
    plt.suptitle(
        'Probability of selecting 1-st option',
        fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/probOfFirstByTrials.png')
    plt.close()
    #%%
    for metric in ['decisionNumber', 'decisionPosition', 'decisionImage']:

        plt.figure(figsize=(35,15))

        for (i, animal_id) in enumerate(ids):
            temp_df = animal_data[animal_data['animalID'] == animal_id].reset_index(drop=True)
            times = temp_df[temp_df['trial'] == 1]['trialStart'].reset_index(drop=True)
            time_diffs = pd.to_datetime(times).diff(1).round('30min')
            pstn = temp_df['Option1'].apply(lambda x: int(x[2]))
            img = temp_df['Option1'].apply(lambda x: int(x[4]))
            plt.subplot(len(ids), 1, i+1)
            if metric == 'decisionImage':
                plt.plot(temp_df['trialTotal'], img, color='red', lw=10, alpha=0.4)
            elif metric == 'decisionPosition':
                plt.plot(temp_df['trialTotal'], pstn, color='red', lw=10, alpha=0.4)
            plt.plot(
                temp_df['trialTotal'],
                temp_df[metric],
                'o')
            for (t, ses_start) in enumerate(temp_df['trialTotal'][temp_df['trial'] == 1]):
                plt.axvline(ses_start, color='black')
                plt.text(ses_start, animal_data[metric].max() + 0.1, time_diffs[t])
            plt.title(animal_id, fontweight='bold', fontsize=14)
            plt.xlim([0, max_trial])

        plt.xlabel('Trial')
        plt.suptitle(metric, fontweight='bold', fontsize=16)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plots_path}/{metric}ByTrials.png')
        plt.close()
    #%%
    for metric in ['decisionNumber', 'decisionPosition', 'decisionImage']:

        decision_counts = animal_data.groupby(['animalID', metric], as_index=False)['trial'].count()

        plt.figure(figsize=(10,6))
        sns.barplot(data=decision_counts, x='animalID', y='trial', hue=metric)
        plt.ylabel('Times Chosen')
        plt.title(f'# {metric} Over All Trials', fontweight='bold', fontsize=16)
        plt.legend(bbox_to_anchor=(1.01,1), loc="upper left", title='Option')
        plt.savefig(f'{plots_path}/{metric}Total.png')
        plt.close()
    #%%
    opt_choice_df = animal_data.groupby(by=['animalID', 'session'], as_index=False)[['optimal_choice']].mean()

    plt.figure(figsize=(15,15))

    for (i, animal_id) in enumerate(ids):
        temp_df = opt_choice_df[opt_choice_df['animalID'] == animal_id]
        plt.subplot(len(ids), 1, i+1)
        plt.plot(
            temp_df['session'],
            temp_df['optimal_choice'],
            'o-')
        plt.title(animal_id, fontweight='bold', fontsize=14)
        plt.xlim([0.5, opt_choice_df.session.max() + 0.5])
        plt.ylim([-0.1, 1.1])

    plt.xlabel('Session')
    plt.suptitle('Ratio of Optimal Choice by Session', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/OptChoiceRatioBySessions.png')
    plt.close()
    #%%
    plt.figure(figsize=(15,15))

    for (i, animal_id) in enumerate(ids):
        temp_df = animal_data[animal_data['animalID'] == animal_id]
        temp_df = temp_df \
            .groupby(by=['session'], as_index=False)[['trial']] \
            .max()
        plt.subplot(len(ids), 1, i+1)
        plt.plot(
            temp_df['session'],
            temp_df['trial'],
            'o-')
        plt.title(animal_id, fontweight='bold', fontsize=14)
        plt.xlim([0.5, opt_choice_df.session.max() + 0.5])
        # plt.ylim([0, 50])

    plt.xlabel('Session')
    plt.suptitle('Number of trials by Session', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/nTrialsBySessions.png')
    plt.close()
    #%%
    plt.figure(figsize=(15,15))

    for (i, animal_id) in enumerate(ids):
        temp_df = animal_data[animal_data['animalID'] == animal_id]
        temp_df = temp_df \
            .groupby('session', as_index=False) \
            .agg({'trialStart': 'min', 'trialEnd': 'max'})
        time_diff = pd.to_datetime(temp_df['trialEnd']) - pd.to_datetime(temp_df['trialStart'])
        time_diff = time_diff.dt.total_seconds() / 60
        plt.subplot(len(ids), 1, i+1)
        plt.plot(
            temp_df['session'],
            time_diff,
            'o-')
        plt.title(animal_id, fontweight='bold', fontsize=14)

    plt.xlabel('Session')
    plt.suptitle('Duration of Session', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_path}/SessionDuration.png')
    plt.close()

    print("Done!")

