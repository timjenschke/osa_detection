import mne
import os
import pandas as pd
import numpy as np
from scipy import signal 
import pyarrow
import gc

from oxygen import Oxygen
from ecg import ECG



class Patient:

    def __init__(self, id, file_path):
        """
        Creates for a patient a dataframe that lists for every episode of 1-min a set of 150 features generated from SpO2 and ECG data
        obtained in a PSG examination. 

        Dataframe will be stored as parquet file with a filename like "sub-XXXXXXXXXXXXXXX_ses-1_feature_dataframe.parquet"
        
        Parameters
            id : str - Identifer of the subject, should look like: "sub-XXXXXXXXXXXXXXX"
            file_path : str - Folder path, where subjects are stored, should end on site identifier, like: ".../.../I0002"

        """

        self.id = id # sub-I0002150000163 
        self.file_path = file_path # "C:\Users\timje\Documents\dataset\annotation_files3\  (sub-I0002150000163\ses-1\eeg\)"
        self.file_paths_feature_dataframes = [] # pathways where feature dataframe are stored

        folders = [f for f in os.scandir(os.path.join(self.file_path, self.id)) if f.is_dir()] #hier path join eigefügt
        folder_names = [s.name for s in folders]

        for i, session in enumerate(folder_names):

            # file paths for retrieving and storing data 
            file_data = os.path.join(self.file_path, self.id, f"{session}", "eeg", f"{self.id}_{session}_task-PSG_eeg.fif")
            file_annotations = os.path.join(self.file_path, self.id, session, "eeg", f"{self.id}_{session}_task-psg_events_annotations.csv")
            file_feature_dataframe = os.path.join(self.file_path, self.id, f"{session}", "eeg", f"{self.id}_{session}_feature_dataframe.parquet")

            # extract data and preprocess it 
            raw_ecg, raw_spo2, timescale = self.extract_psg_data(file_data)

            if timescale[1] == 0.002:
                self.fs = 500
            elif timescale[1] == 0.005:
                self.fs = 200
            else:
                raise ValueError("Sampling frequency is neither 200 nor 500 Hz.")

            clean_ecg, times_ecg, clean_spo2, times_spo2 = self.preprocess_signals(raw_ecg, raw_spo2, timescale, self.fs)

            # get the masks: True if epsiode usable for training
            artefact_mask_ecg = self.artefact_detection_ecg(clean_ecg, self.fs)
            artefact_mask_spo2 = self.artefact_detection_spo2(clean_spo2)

            # True for episodes containg apnea 
            apnea = self.derive_apnea_events_in_episodes(file_annotations, len(artefact_mask_ecg))

            # generate features from data and store it at file_feature_dataframe
            self.derive_features_for_classification(clean_ecg, artefact_mask_ecg, clean_spo2, artefact_mask_spo2, apnea, file_feature_dataframe, self.id, self.fs)
            self.file_paths_feature_dataframes.append(file_feature_dataframe)

            # reduce memory 
            del clean_ecg, clean_spo2, times_ecg, times_spo2
            del artefact_mask_ecg, artefact_mask_spo2, apnea
            gc.collect()


    def extract_psg_data(self, file_path):
        """
        Extract SpO2 and ECG data from fif file and return data in numpy arrays.
        
        Parameters
            file_path : string - file path to the fif file containing the ECG and SpO2 signals 

    
        Returns
            ecg : Array - numpy array containing ECG signal as points (no time axis given)
            spo2 : Array - numpy array containing SpO2 signal as points (no time axis given)
            times : Array - numpy array containg the corresponding time points

        """

        raw = mne.io.read_raw_fif(file_path, preload=True)

        ecg = raw.get_data(picks=['EKG'])[0] 
        spo2 = raw.get_data(picks=['SaO2'])[0]

        times = raw.times

        return ecg, spo2, times


    def artefact_detection_ecg(self, ecg_clean, fs):
        """
        Algorithm based on autocorrelation to detect corrupted ECG episodes that rejects the 5% 1-min segements most dissimilar towards the other episodes.
        
        Varon, Carolina et al. (2012). “Robust artefact detection in long-term ECG recordings based on autocorrelation function similarity and percentile analysis”. 
        In: 2012 Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE, pp. 3151–3154. DOI: https://doi.org/10.1109/EMBC.2012.6346633

        Parameters
            ecg_clean : Array - ECG signal (no time axis given)
            fs : int - Sampling frequency of the ECG signal
        
        Returns
            mask : Array - Boolean values, FALSE when episode was rejected. 

        """
        episode_length = 60
    
        samples_per_episode = int(episode_length * fs)
        n_episodes = len(ecg_clean) // samples_per_episode
        
        if n_episodes == 0:
            return np.array([], dtype=bool)
        
        # trim signal to fit exact number of episodes
        trimmed_signal = ecg_clean[:n_episodes * samples_per_episode]
        
        # shape (n_episodes, samples_per_episode)
        episodes = trimmed_signal.reshape(n_episodes, samples_per_episode)
        
        # calculate mean and std for each episode (axis=1)
        means = np.mean(episodes, axis=1, keepdims=True)  
        stds = np.std(episodes, axis=1, keepdims=True)    
        
        # handle zero std: add epsilon to avoid division by zero
        stds_safe = np.where(stds > 0, stds, 1.0)
        
        normalized_episodes = (episodes - means) / stds_safe 
        
        fft_length = 2 * samples_per_episode
        
        # FFT of all episodes at once
        fft_episodes = np.fft.fft(normalized_episodes, n=fft_length, axis=1)
        
        power_spectrum = np.abs(fft_episodes) ** 2
        
        # inverse FFT to get autocorrelation
        autocorr_full = np.fft.ifft(power_spectrum, axis=1).real
        
        # take only the positive values
        autocorr = autocorr_full[:, :samples_per_episode]
    
        # L2 norm of each autocorrelation vector
        row_norms = np.linalg.norm(autocorr, axis=1, keepdims=True)
        row_norms_safe = np.where(row_norms > 0, row_norms, 1.0)
        norm_autocorr = autocorr / row_norms_safe  
        
        # cosine similarity between all pairs of autocorrelation vectors
        similarity_matrix = np.dot(norm_autocorr, norm_autocorr.T) 
        
        # sum up columns to get weights 
        weights = np.sum(similarity_matrix, axis=0)
        
        # reject smallest 5% of the weights --> artefacts
        threshold = np.percentile(weights, 5)               
        mask = weights > threshold
        
        return mask
        

    def artefact_detection_spo2(self, spo2_clean):
        """
        Algorithm to detect corrupted SpO2 epsiodes. Rejects if any value in 1-min segment is below 66%. 

        Parameters
            spo2_clean : Array - SpO2 signal (no time axis given) sampled at 1 Hz
        
        Returns
            mask : Array - Boolean values, FALSE when episode was rejected

        """
        episode_length = 60

        n_chunks = len(spo2_clean) // episode_length # ignore the last (<60)samples
        trimmed_spo2_downsampled = spo2_clean[:n_chunks*episode_length]
        chunks = trimmed_spo2_downsampled.reshape(n_chunks,episode_length)
        drops_per_chunk = np.sum(chunks < 0.66, axis=1)
        mask = drops_per_chunk == 0

        return mask

    
    def preprocess_signals(self, ecg_raw, spo2_raw, times, fs):
        """
        Clean ECG and downsample SpO2 signal. ECG values are filtered with a Butterworth bandpass filter between 0.5-48 Hz.
        SpO2 are downsampled to 1 Hz and values outside 0-100% are prevented (happenend due to downsampling).

        Parameters
            ecg_raw : Array - ECG signal (no time axis given)
            spo2 : Array - SpO2 signal (no time axis given)
            times : Corrsponding time axis for ECG and SpO2 signal

        Returns
            ecg_clean : Array - ECG signal filtered with Butterworth bandfilter between 0.5-48 Hz
            ecg_times : Array - New time scale for ECG signal.
            spo2_downsampled : Array - SpO2 signal downsampled to 1 Hz.
            times_new_spo2 : Array - New time scale for SpO2 signal           

        """

        # ----------- SpO2 -----------

        signal_duration = len(spo2_raw) 
        samples_decimated = signal_duration // fs  
    
        spo2_downsampled = signal.decimate(spo2_raw,fs, ftype="fir")
        
        # prevent values outside 0-100%
        spo2_downsampled = np.clip(spo2_downsampled, 0.0, 1.0)
        
        # get adapted timescale
        indeces = np.linspace(0, signal_duration, samples_decimated, endpoint=False).astype(int) #get indeces of position of time in times
        times_new_spo2 = times[indeces] #get values of times at indeces

        # ----------- ECG ----------- 
        b_hp, a_hp = signal.butter(N=3, Wn=[0.5,48.0], btype='bandpass', fs=fs)
        ecg_clean = signal.filtfilt(b_hp, a_hp, ecg_raw)
        ecg_times = times
        
        return ecg_clean, ecg_times, spo2_downsampled, times_new_spo2 
        

    def derive_features_for_classification(self, ecg_data, ecg_mask, spo2_data, spo2_mask, apnea, file_path, id, fs):
        """
        Calculates all features for later ML classification. Stores them as a dataframe in a parquet file on disk.
        
        Parameters
            ecg_data : Array - ECG signal
            ecg_mask : Array - Boolean mask marking ECG segments of 1-min that should not be considered
            spo2_data : Array - SpO2 signal
            spo2_mask : Array - Boolean mask marking SpO2 segments of 1-min that should not be considered.
            apnea : Array - Mask identifying segments as Apnea, Hypopnea or Normal episodes.
            file_path : str - File path (parquet file) where dataframe with features for segment of 1-min should be stored.

        Returns
            None

        """

        df_ecg = ECG(ecg_data, ecg_mask, fs).get_features() # ECG keep sampling frequency 
        df_spo2 = Oxygen(spo2_data, spo2_mask).get_features() # SpO2 changed fs to 1 Hz
        df_apnea = pd.DataFrame(apnea, columns=["Apnea"])

        metafile = pd.read_csv('C:\\Users\\timje\\Documents\\dataset\\I0002_psg_metadata_2025-09-08.csv')
        ahi_file = pd.read_csv("C:\\Users\\timje\\Documents\\dataset\\I0002_apnea_event_per_subject_SELFMADE.csv")

        column = metafile[metafile['BIDSFolder'] == id]
        age = pd.DataFrame(np.full(len(apnea), column['AgeAtVisit'].values[0]), columns=["Age"])
        sex = pd.DataFrame(np.full(len(apnea), column['SexDSC'].values[0]), columns=["Sex"])

        subject_id = pd.DataFrame(np.full(len(apnea), id), columns=["Subject"])


        column = ahi_file[ahi_file['BIDSFolder'] == id]
        ahi = pd.DataFrame(np.full(len(apnea), column['AHI'].values[0]), columns=["AHI"])

        all_features = pd.concat([df_apnea, subject_id, age, sex, ahi, df_spo2, df_ecg], axis=1)

        # store feature dataframe on disk
        all_features.to_parquet(file_path)

        print(f"Features stored at {file_path}")

    

    def derive_apnea_events_in_episodes(self, file_path, number_of_episodes):
        """
        
        Derives in which episode of 60s apena/hypopnea events occur. 
        
        Returns mask indicating Normal, Apnea or Hypopnea for each segment. As the Human Sleep Project was evaluated in episodes of 30s, pairs 
        of episodes are taken and marked as Apnea if at least one episode contains apnea. If one or both episodes contain hypopnea, the episode
        is marked as Hypopnea. If both occur, the episode will be marked as apnea, as that is clinically more relevant. 

        Parameters
            file_path : str - File path of the sub-XXXXXXXXXXXXXX_ses-1_task-psg_events_annotations file
            number_of_episodes : int - Number of segments of 1-min that exist for subject

        Returns
            - : list - List of values indicatiing Normal, Apnea or Hypopnea episode 


        """

        df = pd.read_csv(file_path)

        group_id =  df["Epoch"] // 2 #np.arange(len(df)) // 2

        is_apnea = df['Type'] == 'Apnea'
        is_hypopnea = df['Type'] == 'Hypopnea'

        has_apnea = is_apnea.groupby(group_id).transform('max')
        has_hypopnea = is_hypopnea.groupby(group_id).transform('max')

        classification = pd.Series('Normal', index=df.index)
        classification.loc[has_hypopnea] = 'Hypopnea'
        classification.loc[has_apnea] = 'Apnea'

        # reduces the dataframe from 30s -> 60s (Length N -> Length N/2)
        new_df = pd.DataFrame({
            'Epoch': group_id,
            'Classification': classification
        }).drop_duplicates(subset=['Epoch']) # This removes the duplicates!

        # create the master table for all episodes
        apnea_in_episodes = pd.DataFrame(np.arange(number_of_episodes), columns=["Epoch"])

        apnea_in_episodes = pd.merge(apnea_in_episodes, new_df, on='Epoch', how='left')

        apnea_in_episodes['Classification'] = apnea_in_episodes['Classification'].fillna('Normal')

        return list(apnea_in_episodes['Classification'])


    



