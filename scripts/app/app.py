import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title='Activity Dashboard',
    layout="centered")
# st.title('Q-Learning Model Demo')

data_file = st.sidebar.file_uploader("Cleaned data file:")

if data_file is not None:
    animal_data = pd.read_csv(data_file)

    st.sidebar.select_slider("Min trial", options=range(1, animal_data['session'].max() -1 ))
    st.sidebar.select_slider("Max trial", options=range(1, animal_data['session'].max()))

    ids = animal_data['animalID'].unique()
    animal_data.sort_values(by=['animalID', 'session', 'trial'], inplace=True)

    animal_data.insert(3, 'trialTotal', np.NaN)
    animal_data['trialTotal']  = animal_data.groupby('animalID').cumcount()+1

    animal_data.insert(4, 'trialByScenario', np.NaN)
    animal_data['trialByScenario']  = animal_data.groupby(['animalID', 'scenario']).cumcount()+1

    max_trial = animal_data.trialTotal.max()

    for metric in ['decisionLatency', 'rewardLatency', 'startLatency']:

        plt.figure(figsize=(15,30))

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
        plt.show()

        st.pyplot(plt)