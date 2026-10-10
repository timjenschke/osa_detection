import os
import boto3
import mne
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import numpy as np
import pandas as pd

"""
Description 

This script downloads a selected group of subjects from cohort I0002 of the Human Sleep Proejct.
https://bdsp.io/content/hsp/3.0/  

The methods create_subject_set_for_download() is responsbile for subject selection. 
Before usage replace bucket name with your alias. 

"""


# --- SETTINGS ---
BUCKET_NAME = "bdsp-credentialed-ac-PRIVAT" #replace with your own credentials
SOURCE_PREFIX = "PSG/bids/" # Changed to root BIDS folder to find multiple subjects
DEST_LOCAL_PATH = r"C:\Users\timje\Documents\dataset\balanced_data"
PATTERNS = ["psg_events_annotations", "psg_sleep_annotations", "task-PSG_eeg.e"]
# ----------------

subjects = []

#-----------------
# this decorator tells the function to retry if a network/AWS error occurs.
# it will wait exponentially (1s, 2s, 4s...) and stop after 5 attempts
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ClientError, BotoCoreError, ConnectionError)),
    before_sleep=lambda retry_state: print(f"Connection lost. Retrying... (Attempt {retry_state.attempt_number})")
)
def safe_s3_download(s3_client, bucket, key, local_path):
    s3_client.download_file(bucket, key, local_path)

def extract_ecg_sao2_from_edf(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        # Use a try-except or check if channels exist to prevent crashing on bad files
        channels = [ch for ch in ["EKG", "SaO2"] if ch in raw.ch_names]
        if not channels:
            print(f"Warning: No EKG/SaO2 found in {file_path}. Skipping extraction.")
            return
            
        raw_picked = raw.pick(channels)
        new_path = os.path.splitext(file_path)[0] + ".fif"
        raw_picked.save(new_path, overwrite=True)
        print(f"Successfully stored as .fif")
    except Exception as e:
        print(f"MNE Extraction failed for {file_path}: {e}")

def download_s3_annotations():
    s3 = boto3.client('s3')
    
    print("Step 1: Listing files...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=SOURCE_PREFIX)
        
        processed_subjects = set()
        total_files_downloaded = 0
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                s3_key = obj['Key']
                file_name = os.path.basename(s3_key)
                
                # identify the subject from the path (e.g., 'sub-I0002')
                path_parts = s3_key.split('/')
                subject_id = next((part for part in path_parts if part.startswith('sub-')), None)
                
                if not subject_id:
                    continue

                # only download subjects that are in the list to download
                if subject_id not in subjects:
                    continue 
                else:
                    print(f"Start downloading {subject_id}.")

                # filter by patterns
                if any(pattern in file_name for pattern in PATTERNS):
                    relative_path = os.path.relpath(s3_key, SOURCE_PREFIX)
                    local_file_path = os.path.join(DEST_LOCAL_PATH, relative_path)
                    
                    local_folder = os.path.dirname(local_file_path)
                    os.makedirs(local_folder, exist_ok=True)
                    
                    try:
                        # Use the retry-enabled download function
                        safe_s3_download(s3, BUCKET_NAME, s3_key, local_file_path)
                        total_files_downloaded += 1
                        
                        if local_file_path.lower().endswith(".edf"):
                            extract_ecg_sao2_from_edf(local_file_path)
                            if os.path.exists(local_file_path):
                                os.remove(local_file_path)
                                
                    except Exception as e:
                        print(f"Failed to download {file_name} after retries: {e}")
                        continue


    except Exception as e:
        print(f"A critical error occurred: {e}")


def create_subject_set_for_download():
    """This method creates a list of subject identifiers.
    The severity group sizes of OSA are set to 200. Furthermore the amount of women in each group is chosen with 50%, 
    thus the dataset will roughly be equal in sizes with respect to severity classes and gender.
    The subjects are chosen randomnly, however being biased towards subjects younger than 50. The probabilities are doubled.
    
    """
    METADATA_CSV_I0002 = "C:/Users/timje/Documents/dataset/I0002_psg_metadata_2025-09-08.csv"
    CSV_AHI = "C:/Users/timje/Documents/dataset/I0002_apnea_event_per_subject_SELFMADE.csv"

    NORMAL = 200
    MILD = 200
    MODERATE = 200
    SEVERE = 200
    WOMEN_RATE = 0.5
    SUBJ_SELECTED = []

    rng = np.random.default_rng(seed=42)

    dataframe_ahi = pd.read_csv(CSV_AHI)
    dataframe_metadata = pd.read_csv(METADATA_CSV_I0002)

    # select only first sessions (as we do not know if some therapeutic measures were applied at later sessions) and thereby make sure that subject id is unique
    dataframe_ahi = dataframe_ahi[dataframe_ahi["SessionID"] == 1]

    # make sure that sleep event anotations exists 
    subjects = dataframe_ahi["BIDSFolder"]
    for subject in subjects:
        event_file_exits = dataframe_metadata.loc[(dataframe_metadata["BIDSFolder"] == subject) & (dataframe_metadata["SessionID"] == 1), "EventAnnotations"].values[0]
        if event_file_exits != "Y":
            # delete row by keeping all values except the subject row
            dataframe_ahi = dataframe_ahi[dataframe_ahi["BIDSFolder"] != subject]

    for apnea in [(0.0, 5.0, NORMAL), (5.0, 15.0, MILD), (15.0,30.0, MODERATE), (30.0, 1000.0, SEVERE)]:
        lower, upper, severity = apnea

        df_male = dataframe_ahi[(dataframe_ahi["AHI"] > lower) & (dataframe_ahi["AHI"] < upper) & (dataframe_ahi["SexDSC"] == "M")]
        sbj_male = df_male["BIDSFolder"].tolist()
        mask_50 = np.where(df_male['AgeAtVisit'] <= 50, 2,1)
        prob = mask_50 / sum(mask_50) # normalized to values between 0 and 1 add adds up to exactly 1
        #prefer males which are 50 or younger (doubled probability)
        random_subj_male = rng.choice(sbj_male, size=int(severity*(1-WOMEN_RATE)), replace=False, p=prob)

        df_female = dataframe_ahi[(dataframe_ahi["AHI"] > lower) & (dataframe_ahi["AHI"] < upper) & (dataframe_ahi["SexDSC"] == "F")]
        sbj_female = df_female["BIDSFolder"].tolist()
        mask_50 = np.where(df_female['AgeAtVisit'] <= 50, 2,1)
        prob = mask_50 / sum(mask_50)
        #prefer females which are 50 or younger (doubled probability)
        random_subj_female = rng.choice(sbj_female, size=int(severity*WOMEN_RATE), replace=False, p=prob)

        SUBJ_SELECTED.extend(random_subj_male.tolist())
        SUBJ_SELECTED.extend(random_subj_female.tolist())

    return SUBJ_SELECTED

if __name__ == "__main__":
    subjects = create_subject_set_for_download()
    download_s3_annotations()