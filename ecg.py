from scipy import signal
from scipy.interpolate import CubicSpline
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from ecgdetectors import Detectors #10.5281/zenodo.3353396 
from hrv import HRV #10.5281/zenodo.3353396 
import pywt # https://joss.theoj.org/papers/10.21105/joss.01237 zitieren!

class ECG:

    def __init__(self, ecg_data, mask, fs):
        """
        Generates 39 SpO2 features per 60s episodes from ECG signal.

        Parameters
            ecg_data : Array - ECG signal
            mask : Array - Boolean values indicating corrupted ECG segment of 1-min
            fs : int - Sampling frequency of the ECG signal
    
        """
        self.ecg = ecg_data
        self.fs = fs
        self.mask = mask

        if self.fs not in [200,500]:
            raise ValueError("Sampling frequency neither 200 nor 500 Hz.")

        self.features_per_episode = self.generate_ecg_features(self.ecg, self.fs, self.mask)

    @staticmethod
    def edr(ecg, org_signal_fs, fs=2, window_size=25):

        """ 
        Calculates ECG-derived respiration (EDR) using Pan-Tomkins algorithm for R-peak detection. Returns respiration wave smoothed by cubic spline interpolation with a frequency of fs.

        Arunachalam, Shivaram P. and Lewis F. Brown (2009). “Real-time estimation of the ECG-derived respiration (EDR) signal using a new algorithm for baseline wander
        noise removal”. In: 2009 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, pp. 5681–5684. DOI: https://doi.org/10.1109/IEMBS.2009.5333113

        Parameter
            ecg : Array - ECG signal
            org_signal_fs : int - Sampling frequency of ECG
            fs : int - Sampling frequency of EDR 
            window_size : Neighborhood around detected R-peak that should be checked for real R 
        
        Return
            edr_wave_clean : Array - EDR wave sampled as fs frequency 
        """

        # adapt window size for higher sampling frequency (more data points)
        if org_signal_fs != 200:
            window_size = 50

        # detect R peaks and find there corresponding amplitudes
        detectors = Detectors(sampling_frequency=org_signal_fs)
        r_peaks = np.array(detectors.pan_tompkins_detector(ecg))


        #amplitudes = ecg[r_peaks]
        # correction of R-peaks: these algorithms usually do not find exact peaks, thus amplitudes will be disturbed (even negative: close to S point)
        # account for it, be searching close to peaks for highest value = R point
        
        offsets = np.arange(-window_size, window_size + 1)
        neighborhoods = r_peaks[:, None] + offsets #creates 2D array, with all numerical neighbors, so for 20, it would be [17,18,19,20,21,22,23]

        # make sure that the neighbors are not below 0 or above the length of ecg episode
        neighborhoods = np.clip(neighborhoods, 0, len(ecg)-1).astype(int)

        neighborhood_values = ecg[neighborhoods] # get the corresponding ecg values;

        # more precise amplitudes of the R-peaks
        amplitudes = np.max(neighborhood_values, axis=1)

        amplitudes_averaged = amplitudes

        # fix timescaling due to manipulated amplitudes
        max_indices = np.argmax(neighborhood_values, axis=1)
        r_peaks_new =  neighborhoods[np.arange(len(neighborhoods)), max_indices] # get in each row the maximum value

        r_peaks_s_new = r_peaks_new / org_signal_fs

        # apply cubic spline interpolation to get smooth respiration curve
        # and further assure uniform sampling (r peaks are not uniform) 
        cs = CubicSpline(r_peaks_s_new, amplitudes_averaged, bc_type='natural') 
        edr_time = np.linspace(r_peaks_s_new[0], r_peaks_s_new[-1], int(r_peaks_s_new[-1]*fs)) 
        edr_wave = cs(edr_time)

        # apply butterworth filter in the range of physiological brething ranges (6-30/min or 0.1-0.5 Hz)
        b_hp, a_hp = signal.butter(N=3, Wn=[0.1,0.5], btype='bandpass', fs=fs)
        edr_wave_clean = signal.filtfilt(b_hp, a_hp, edr_wave)

        """Plotting for sanity checks
        plt.figure(figsize=(25, 8))
        plt.plot(r_peaks_s_new, amplitudes, 'o', label='R peaks', color='red')
        plt.plot(edr_time, edr_wave_clean, label='Cubic Spline', color='blue')
        plt.plot(np.linspace(0,60,12000), ecg, label="ECG", color="black")
        plt.legend()
        plt.grid(True)
        plt.show()
        """

        return edr_wave_clean

    @staticmethod
    def pds_fft(data):

        """
        Calculates 256 point FFT power spectral density (PSD) estimate and keps only first 32 (0.25 Hz) points in relation to ANS frequency range 0.003-0.4 Hz.

        Parameter
            data : Array - ECG signal of 1-min
        
        Return
            ans_psd : Array - First 32 points of the power spectrum
            {sums} : float  - Summed powers of frequency bands VLF, LF and HF

        """
        # use first 256 points to calculate FFT: as there are only 120 samples (2Hz, 60s), zero-pad it --> using 256 points seems to improve resolution
        segment = data[:256] if len(data) >= 256 else np.pad(data, (0, 256 - len(data)))
        fft_vals = np.fft.fft(segment)
        psd = np.abs(fft_vals)**2  #calcualtion of power spectrum energy
        psd_half = psd[:128] #only positive values
        ans_psd = psd_half[:32] # use only the first 32 values: frequency resolution = fs/N = 2/256 = 0.0078 Hz/bin --> 32 bins * 0.0078 = 0.25 Hz (ANS frequency)

        res = 2 / 256 # fs/N
        vlf_band = ans_psd[int(0.003/res) : min(int(0.04/res),32)]
        lf_band = ans_psd[int(0.04/res) :  min(int(0.15/res),32)] 
        hf_band = ans_psd[int(0.15/res) : min(int(0.4/res),32)]

        return ans_psd, np.sum(vlf_band), np.sum(lf_band), np.sum(hf_band)

    @staticmethod
    def flatten_numpy_columns(df):
        """
        Finds columns containing numpy arrays and expands them into 
        multiple columns with naming: 'columnName_0', 'columnName_1', etc.
        """
        result_df = df.copy()
        
        cols_to_flatten = []
        for col in df.columns:
            # Use dropna() to find the first actual value to check its type
            non_null_vals = df[col].dropna()
            if not non_null_vals.empty:
                first_val = non_null_vals.iloc[0]
                if isinstance(first_val, np.ndarray):
                    cols_to_flatten.append(col)
        
        for col in cols_to_flatten:
            
            lengths = [len(x) if isinstance(x, np.ndarray) else 0 for x in df[col]]
            max_len = max(lengths)
            
            padded_data = []
            for x in df[col]:
                if isinstance(x, np.ndarray):
                    pad_size = max_len - len(x)
                    # Pad with zeros at the end (0 before, pad_size after)
                    padded_x = np.pad(x, (0, pad_size), mode='constant', constant_values=0)
                    padded_data.append(padded_x)
                else:
                    # If the cell is NaN/None, create an array of zeros
                    padded_data.append(np.zeros(max_len))
            
            matrix = np.array(padded_data)
           
            new_col_names = [f"{col}_{i}" for i in range(matrix.shape[1])]
            expanded_df = pd.DataFrame(matrix, columns=new_col_names, index=df.index)
            
            result_df = pd.concat([result_df, expanded_df], axis=1).drop(columns=[col])
            
        return result_df


    def plot_peak_check(self, chunk, r_peaks, chunk_index):
        """ 
        Plots the ECG signal and marks the detected R-peaks to verify if the detector is missing beats.
        """
        plt.figure(figsize=(15, 4))
        plt.plot(chunk, color='black', linewidth=0.7, label='ECG Signal')
        
        if len(r_peaks) > 0:
            plt.plot(r_peaks, chunk[r_peaks], 'ro', markersize=5, label='Detected R-peaks')
        
        plt.title(f"Chunk {chunk_index} | Peaks Found: {len(r_peaks)}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()




    def generate_ecg_features(self, cleaned_ecg, fs, mask):
        """
        Generate 111 feature from ECG signal. 

        Bsoul, Majdi, Hlaing Minn, and Lakshman Tamil (2011). “Apnea MedAssist: Real-time Sleep Apnea Monitor Using Single-Lead ECG”.
        In: IEEE Transactions on Information Technology in Biomedicine 15.3, pp. 416–427. DOI: https://doi.org/10.1109/TITB.2010.2087386

        Parameter
            cleaned_ecg : Array - ECG signal 
            fs : int - Sampling frequency of ECG
            mask : Array - Boolean values indicating corrupted ECG segments of 1-min

        Return 
            df_spreaded : DataFrame - Dataframe containing calculated features for each segment of 1-min
        """

        detectors = Detectors(sampling_frequency=fs)
        hrv = HRV(sampling_frequency=fs)

        n_chunks = len(cleaned_ecg) // int(60*fs)       
        trimmed_ecg = cleaned_ecg[:int(n_chunks*fs*60)]       
        chunks = np.reshape(trimmed_ecg, (n_chunks,int(60*fs)))   

        chunks_and_features = []  

        for i, chunk in enumerate(chunks):

            if mask[i] == True:

                r_peaks = np.array(detectors.pan_tompkins_detector(chunk)) # [63, 277, 459, 640, 812, 1000,..., 11861] first round (these are *indices* of R peaks)
                r_peaks_s = r_peaks / fs

                if len(r_peaks) < 40: 
                    print(f"Chunk {i} has too few peaks ({len(r_peaks)}). Skipping calculations and marking as False.")
                    mask[i] = False


                else:
                    rr = np.diff(r_peaks_s)

                    rr_mean = np.mean(rr)
                    rr_sd = np.sqrt(np.var(rr))

                    nn50 = hrv.NN50(r_peaks)
                    nn20 = hrv.NN20(r_peaks)

                    pNN50 = nn50 / len(rr)
                    pNN20 = nn20 / len(rr)

                    #standard deviation of interbeat differntials
                    sdsd = hrv.SDSD(r_peaks) 

                    # root mean square of interbeat differntials 
                    rmssd = hrv.RMSSD(r_peaks)

                    # serial correlation coefficients (+ 1e-9 to prevent dividing by 0)
                    scc1 = np.sum((rr[:-1] - rr_mean) * (rr[1:] - rr_mean)) / np.sum((rr-rr_mean)**2 + 1e-9)
                    scc2 = np.sum((rr[:-2] - rr_mean) * (rr[2:] - rr_mean)) / np.sum((rr-rr_mean)**2 + 1e-9)
                    scc3 = np.sum((rr[:-3] - rr_mean) * (rr[3:] - rr_mean)) / np.sum((rr-rr_mean)**2 + 1e-9)
                    scc4 = np.sum((rr[:-4] - rr_mean) * (rr[4:] - rr_mean)) / np.sum((rr-rr_mean)**2 + 1e-9)
                    scc5 = np.sum((rr[:-5] - rr_mean) * (rr[5:] - rr_mean)) / np.sum((rr-rr_mean)**2 + 1e-9)

                    #fractal Alan factors 
                    bins5 = (r_peaks_s // 5).astype(int) # [0, 0, 1, 1, 1, 2,...]
                    hb_bins5 = np.bincount(bins5)
                    at5 = np.average(np.diff(hb_bins5)**2) / np.average(hb_bins5)*2

                    bins10 = (r_peaks_s // 10).astype(int) 
                    hb_bins10 = np.bincount(bins10)
                    at10 = np.average(np.diff(hb_bins10)**2) / np.average(hb_bins10)*2

                    bins15 = (r_peaks_s // 15).astype(int)
                    hb_bins15 = np.bincount(bins15)
                    at15 = np.average(np.diff(hb_bins15)**2) / np.average(hb_bins15)*2

                    bins30 = (r_peaks_s // 30).astype(int)
                    hb_bins30 = np.bincount(bins30)
                    at30 = np.average(np.diff(hb_bins30)**2) / np.average(hb_bins30)*2

                    # Number of extreme points (NEP): https://ieeexplore.ieee.org/document/4352373 
                    diff_rr = np.diff(rr)
                    products = diff_rr[:-1] * diff_rr[1:]
                    nep = np.sum(products < 0) / (len(diff_rr)-2)

                    # edr calculations; in the paper sliding moving average winodw applied to average out outliers
                    # "Both of the irregularly sampled time series RR(m) and EDR(q) are spline-resampled at 2 Hz and then detrended by removing the resampled time-series average."
                    edr_signal = self.edr(chunk, org_signal_fs=fs) # based on s not ms
                    edr_signal = edr_signal - np.mean(edr_signal)
                    #peaks, properties = signal.find_peaks(edr, distance=2) # assumuption: maximum breathing frequenc is 30, thus peaks must be at least 2 seconds apart

                    edr_mean = np.mean(edr_signal) # sampling frequency should he sufficient high (5 Hz; paper uses 2 Hz) to approach mean if it would be continuous = ssampling frequency has no influence on mean
                    edr_sd = np.sqrt(np.var(edr_signal))


                    # -------------spectral features -----------------

                    # "Both of the irregularly sampled time series RR(m) and EDR(q) are spline-resampled at 2 Hz and then detrended by removing the resampled time-series average."
                    spline_fs = 2
                    cs = CubicSpline(np.cumsum(rr), rr, bc_type='natural') # if rr = [0.8, 0.9, 1.2] than np.cumsum(rr) = [0.8, 1.7, 2.9]
                    rr_time_uniform = np.linspace(r_peaks_s[0], r_peaks_s[-1], int(r_peaks_s[-1]*spline_fs))
                    rr_wave_uniform = cs(rr_time_uniform)
                    rr_wave_uniform = rr_wave_uniform - np.mean(rr_wave_uniform)

                    # discrete / decimal wavelet transform
                    coeffs_decimatedWT_RR = pywt.wavedec(rr_wave_uniform, 'db4', level=8)
                    coeffs_decimatedWT_EDR = pywt.wavedec(edr_signal, 'db4', level=9)
                    
                    var_DWT_RR = np.array([np.sum((x-np.mean(x))**2) for x in coeffs_decimatedWT_RR[1:-1]])
                    var_DWT_EDR = np.array([np.sum((x-np.mean(x))**2) for x in coeffs_decimatedWT_EDR[1:-1]])

                    # discrete wavelet transform VLF, HF, LF
                    hf_DWT_RR = np.var(np.concatenate([coeffs_decimatedWT_RR[-2], coeffs_decimatedWT_RR[-3]])) #d2: 0.25 - 0.5 Hz, d3: 0.125  - 0.25 Hz --> contain hf range
                    lf_DWT_RR = np.var(np.concatenate([coeffs_decimatedWT_RR[-4], coeffs_decimatedWT_RR[-5]])) #d4: 0.0625 - 0.125 Hz, d5: 0.03125 - 0.0625 Hz
                    vlf_DWT_RR = np.var(coeffs_decimatedWT_RR[-6]) # d6: 0.0156 - 0.03125 Hz

                    hf_DWT_EDR = np.var(np.concatenate([coeffs_decimatedWT_EDR[-2], coeffs_decimatedWT_EDR[-3]])) #d2: 0.25 - 0.5 Hz, d3: 0.125  - 0.25 Hz 
                    lf_DWT_EDR = np.var(np.concatenate([coeffs_decimatedWT_EDR[-4], coeffs_decimatedWT_EDR[-5]])) #d4: 0.0625 - 0.125 Hz, d5: 0.03125 - 0.0625 Hz
                    vlf_DWT_EDR = np.var(coeffs_decimatedWT_EDR[-6]) # d6: 0.0156 - 0.03125 Hz

                    # perform FFT power spectral density (PSD) estimate 
                    pds_fft_RR, vlf_fft_RR, lf_fft_RR, hf_fft_RR  = self.pds_fft(rr_wave_uniform)
                    pds_fft_EDR, vlf_fft_EDR, lf_fft_EDR, hf_fft_EDR = self.pds_fft(edr_signal)

            if mask[i] == True:
                new_row = {
                    "rr_mean" : rr_mean, "rr_sd" : rr_sd, "NN50": nn50, "NN20": nn20, "pNN50": pNN50, "pNN20": pNN20,
                    "sdsd": sdsd, "rmssd": rmssd, "scc1": scc1, "scc2": scc2, "scc3": scc3, "scc4": scc4, "scc5": scc5,
                    "at5": at5, "at10": at10, "at15": at15, "at30": at30, "nep": nep, "edr_mean": edr_mean, "edr_sd": edr_sd,
                    "rr_DWT_var": var_DWT_RR, "rr_DWT_var_VLF": vlf_DWT_RR, "rr_DWT_var_LF": lf_DWT_RR, "rr_DWT_var_HF": hf_DWT_RR,
                    "rr_psd_fft32": pds_fft_RR, "rr_psd_fft_vlf": vlf_fft_RR, "rr_psd_fft_lf": lf_fft_RR, "rr_psd_fft_hf": hf_fft_RR,
                    "edr_DWT_var": var_DWT_EDR, "edr_DWT_var_VLF": vlf_DWT_EDR, "edr_DWT_var_LF": lf_DWT_EDR, "edr_DWT_var_HF": hf_DWT_EDR,
                    "edr_psd_fft32": pds_fft_EDR, "edr_psd_fft_vlf": vlf_fft_EDR, "edr_psd_fft_lf": lf_fft_EDR, "edr_psd_fft_hf": hf_fft_EDR
                }

            else:
                new_row = {
                    "rr_mean" : None, "rr_sd" : None, "NN50": None, "NN20": None, "pNN50": None, "pNN20": None,
                    "sdsd": None, "rmssd": None, "scc1": None, "scc2": None, "scc3": None, "scc4": None, "scc5": None,
                    "at5": None, "at10": None, "at15": None, "at30": None, "nep": None, "edr_mean": None, "edr_sd": None,
                    "rr_DWT_var": None, "rr_DWT_var_VLF": None, "rr_DWT_var_LF": None, "rr_DWT_var_HF": None,
                    "rr_psd_fft32": None, "rr_psd_fft_vlf": None, "rr_psd_fft_lf": None, "rr_psd_fft_hf": None,
                    "edr_DWT_var": None, "edr_DWT_var_VLF": None, "edr_DWT_var_LF": None, "edr_DWT_var_HF": None,
                    "edr_psd_fft32": None, "edr_psd_fft_vlf": None, "edr_psd_fft_lf": None, "edr_psd_fft_hf": None
                }

            chunks_and_features.append(new_row)

        df = pd.DataFrame(chunks_and_features)
        df_spreaded= self.flatten_numpy_columns(df)
        #df_spreaded.to_csv("C:/Users/timje/Documents/Studium/bachelor/code/feature_tables/ecg_features.csv", index=True)
        return df_spreaded
        
        
    def get_features(self):
        """Return 2D pandas dataframe with ECG features per episode of 60s. First row contains feature names."""
        return self.features_per_episode

