import numpy as np
import pandas as pd
from scipy import signal

class Oxygen:

    def __init__(self, spo2_data, mask, fs=1):
        """
        Generates 39 SpO2 features per 60s episodes from SpO2 signal. 

        Parameters
            spo2_data : Array - SpO2 data (no time axis given) sampled at 1 Hz
            mask : Array - Boolean values indicating which segments are corrupted
        """
        self.spo2 = spo2_data
        self.fs = fs
        self.mask = mask
        
        self.features_per_episode = self.generate_sao2_features(self.spo2, self.fs, self.mask)


    @staticmethod
    def approximate_entropy(x, m=1, r=None):
        """
        Calculates the Approximate Entropy (ApEn) of a time series.
        
        Hornero, Roberto et al. (2007). “Utility of approximate entropy from overnight pulse oximetry data in the diagnosis of the obstructive sleep apnea syndrome”. 
        In: IEEE Transactions on Biomedical Engineering 54.1, pp. 107–113. DOI: https://doi.org/10.1109/TBME.2006.883821
        
        Parameters
            x : Array - SpO2 signal for one segment
            m : int - The length of the compared sequences
            r : float - The filtering level (threshold)

        Returns
            - : float Approximate Entropy
        """
        x = np.array(x)
        n = len(x)
        
        if r is None:
            r = 0.25 * np.std(x)
        
        def _phi(m_len):

            X = np.array([x[i : i + m_len] for i in range(n - m_len + 1)])

            n_seqs = X.shape[0]
            dist_matrix = np.zeros((n_seqs, n_seqs))

            for i in range(n_seqs):
                for j in range(n_seqs):
                    dist_matrix[i, j] = np.max(np.abs(X[i] - X[j]))
    
            C = np.sum(dist_matrix <= r, axis=1) / n_seqs  #- 1 (count self-matching)
            
            valid = C > 0
        
            return np.mean(np.log(C[valid]))

        return _phi(m) - _phi(m + 1)

    @staticmethod
    def lempel_ziv_complexity(x):
        """
        Calculates Lempel-Ziv complexity.

        Kaspar, Felix and HG Schuster (1987). “Easily calculable measure for the complexity of spatiotemporal patterns”. 
        In: Physical review A 36.2, p. 842. DOI: https://doi.org/10.1103/PhysRevA.36.842

        Parameters
            x : Array - SpO2 signal for one segment

        Returns
            c : float - Lempel-Ziv complexity
        
        """

        median = np.median(x)
        binary_pattern = (x > median).astype(int)
        n = len(binary_pattern)

        i = 0
        c = 1
        l = 1 
        k = 1  
        k_max = 1

        while l + k < n:
        
            if binary_pattern[i+k-1] == binary_pattern[l+k-1]:
                k += 1
                if l + k >= n:
                    c += 1
                    return c         
            else:
                k_max = max(k, k_max)
                i += 1

                if i == l:
                    c += 1
                    l = l + k_max
                    
                    i = 0
                    k = 1
                    k_max = 1 
                else:
                    k = 1
                
        return c 

    @staticmethod
    def central_tendency_measure(x, r):

        """
        Implements central tendency measure with radius r. 

        Alvarez, D et al. (2006). “Nonlinear characteristics of blood oxygen saturation from nocturnal oximetry for obstructive sleep apnoea detection”. 
        In: Physiological Measurement 27.4, pp. 399–412. DOI: https://doi.org/10.1088/0967-3334/27/4/006 
        
        Parameters
            x : Array - SpO2 signal for one segment
            r : float - Radius between 0-1 
        
        Return
            ctm : float - Central tendency measure
        """

        diffs = np.diff(x)
        # diffs[1:] everything but first element/comparison: x[i+2] - x[i+1]
        # diffs[:-1] everyting but last element/commparison: x[i+1] - x[i]
        dist = np.sqrt(diffs[1:]**2 + diffs[:-1]**2)
        deltas = (dist < r).astype(int)

        ctm = np.sum(deltas) / (len(x)-2)


        return ctm

    @staticmethod
    def oxygen_desaturation_index(x):
        """ 
        Implementation of the oxygen desaturation index (odi), which counts the occurences oxygen saturation level falls below 2,3 or 4% below baseline.
        Baseline is set to be the median over the whole segment of 1-min, or 0.90 if that in case that should be below 0.90.

        Burgos, Alfredo et al. (2009). “Real-time detection of apneas on a PDA”. In: IEEE Transactions on Information Technology in Biomedicine 14.4, pp. 995–1002.
        DOI: https://doi.org/10.1109/TITB.2009.2034975.

        Due to missing implementation explaination of how to detect desaturations, the desaturation event detection algorithm as described here was used in adapted form: 
        Deviaene, Margot et al. (2019). “Automatic Screening of Sleep Apnea Patients Based on the SpO2 Signal”. 
        In: IEEE Journal of Biomedical and Health Informatics 23.2, pp. 607–617. DOI: https://doi.org/10.1109/JBHI.2018.2817368


        Parameters
            x : Array - SpO2 signal for one segment 
        
        Returns
            odi2, odi2, odi4 : float - Oxygen desaturation index for 2,3 and 4% desaturations with respect to baseline. 
        
        """

        x_copy = x.copy()
        # Padding by adding the first value 2x to the front, the last value two times to the end (needed to perform convolution on whole array)
        x_copy_padded = np.pad(x_copy, (2, 2), mode='edge') 
    
        window = np.ones(5) / 5 # [1,1,1,1,1] / 5 --> shifted through the whole array (convolution) and gets mutlplied with --> average 
        spo2_filtered = np.convolve(x_copy_padded, window, mode='valid')

        derivative = np.diff(spo2_filtered) 
        abs_derivative = np.abs(derivative)

        # Find peaks in the absolute derivative (points of fastest change)
        peaks, _ = signal.find_peaks(abs_derivative, distance=5)
        # Find local maxima in the SpO2 signal 
        peaks_spo2, _ = signal.find_peaks(spo2_filtered, distance=5)

        odi2, odi3, odi4 = (0,0,0)
        events = []

        # described as 1 min average over last clean episode --> adapted to process segment independently
        baseline = np.median(x) if np.median(x) > 0.90 else 0.90
        
        is_desaturating = False
        start_idx = None


        for idx in peaks:
            # negative derivative = start of a drop = destaturation
            if derivative[idx] < 0 and not is_desaturating:
                is_desaturating = True

                # desaturation event starts with local maximum closest to the beginning of desaturation
                prev_peaks = peaks_spo2[peaks_spo2 <= idx]
                start_idx = prev_peaks[-1] if len(prev_peaks) > 0 else 0
                
            # positive derivative = start of resaturation
            elif derivative[idx] > 0 and is_desaturating:

                next_peaks = peaks_spo2[peaks_spo2 >= idx]
                if len(next_peaks) != 0:
                    next_peak = next_peaks[0]
                else:
                    # in case no peak exist anymore, take the last index of x
                    next_peak = len(x)-1


                # check if really full resaturation, or just a small hill in a longer desaturation phase 
                # if distance longer than 30s small hills are taken as end of desaturation event = asummption that they are from different events
                if x[next_peak] > baseline*0.99 or (next_peak - start_idx) > 30:

                    # if indeed resaturation, end the whole desaturation event 
                    end_idx = idx 
                    desat_period = x[start_idx : end_idx + 1]

                    if len(desat_period) > 0: 
                        
                        if np.any(desat_period <= (baseline*0.98)):
                            odi2 += 1
                        if np.any(desat_period <= (baseline*0.97)):
                            odi3 += 1
                        if np.any(desat_period <= (baseline*0.96)):
                            odi4 += 1
                        
                        # event recording for plotting: not necessary for the algorithm 
                        if  np.any(desat_period < (baseline*0.99)):
                            events.append([start_idx, end_idx])

                    is_desaturating = False

        # if end of signal and still no resaturated, assume that full resaturation (might be a partial resaturation) to not lose to much information
        if is_desaturating:
            end_idx = len(x) - 1
            desat_period = x[start_idx : end_idx + 1]
            if len(desat_period) > 0:
                if np.any(desat_period <= (baseline*0.98)): odi2 += 1
                if np.any(desat_period <= (baseline*0.97)): odi3 += 1
                if np.any(desat_period <= (baseline*0.96)): odi4 += 1
                if np.any(desat_period <= (baseline*0.99)):
                    events.append([start_idx, end_idx])


        odi2 = odi2 / 60
        odi3 = odi3 / 60
        odi4 = odi4 / 60

        """
        if events:
            plt.figure(figsize=(30, 10)) 
            plt.plot(x, color='steelblue', linewidth=1, label='SpO2 Signal')
            for ev in events:
                plt.axvspan(ev[0], ev[1], color='red', alpha=0.2)
            plt.axhline(y=baseline, color='g', linestyle='--', label=f"Baseline of {np.round(baseline,4)}")
            plt.axhline(y=baseline*0.98, color='y', linestyle='--', label="Threshold of odi detection")
            plt.axhline(y=baseline*0.97, color='y', linestyle='--', label="Threshold of odi detection")
            plt.axhline(y=baseline*0.96, color='y', linestyle='--', label="Threshold of odi detection")
            plt.title(f"SpO2 Signal: {len(events)} Events Detected")
            plt.xlabel("Time (samples)")
            plt.ylabel("Oxygen Saturation")
            plt.ylim(0.8, 1.05)
            plt.legend()
            plt.show()

        print(f"Events found: {len(events)}")
        print(baseline)
        """
        return odi2, odi3, odi4


    @staticmethod
    def odixy(data, x:int, y:int):
        """
        Calculates the ODIxy, number of desaturations events that fall at least x % below set basline and 
        stay for at least y seconds below that basline. 

        Parameters:
            data : Array - SpO2 signal for one segment
            x : int - threshold percentage below baseline
            y : int - seconds event needs to last

        Return
            odi : int - odixy
        
        """

        odi = 0
        cutoff = np.percentile(data, q=80) 
        baseline = np.mean(data[data >= cutoff])

        below_x = (data < np.round(baseline*(100-x)/100, 2)).astype(int)
        diff = np.diff(below_x) # +1 if falling below cutoff, 0 if no change, -1 if rising above threshold

        # every occasion (index) SpO2 falls below the baseline by x % 
        events = np.where(diff == +1)[0]

        for event in events:
            # below_x is 1 if SpO2 below baseline: check if after event occured, signals stays below baseline for y seconds
            if np.sum(below_x[event+1:event+y+1]) == y: #event+1 due to shift in diff array
                odi += 1
        
        return odi 


    @staticmethod
    def odis(data):
        """
        Calculates total number of samples that fall below 2,3,4 and 5% below baseline.

        Parameter
            data : Array - SpO2 signal for one segment
        
        Return
            odiS2, odiS3, odiS4, odiS5 : float - Total counts for 2,3,4 and 5% below baseline
        
        """

        cutoff = np.percentile(data, q=0.8)
        baseline = np.mean(data[data >= cutoff])

        odiS2 = np.sum((data < np.round(baseline*0.98, 2)).astype(int))
        odiS3 = np.sum((data < np.round(baseline*0.97, 2)).astype(int))
        odiS4 = np.sum((data < np.round(baseline*0.96, 2)).astype(int))
        odiS5 = np.sum((data < np.round(baseline*0.95, 2)).astype(int))

        return odiS2, odiS3, odiS4, odiS5


    def generate_sao2_features(self, cleaned_sao2, fs, mask):
        """
        Generates 39 oxygen-related features. Generates features for each segment of 1-min and collects them in a dataframe.

        Xie, Baile and Hlaing Minn (2012). “Real-time sleep apnea detection by classifier combination”. 
        In: IEEE Transactions on information technology in biomedicine 16.3, pp. 469–477. DOI: https://doi.org/10.1109/TITB.2012.2188299

        Parameter
            cleaned_sao2 : Array - SpO2 signal 
            fs : int - Sampling frequency of SpO2 signal
            mask : Array - Boolean values indicating corrupted SpO2 segments of 1-min

        Return
            df : DataFrame - Feature for each segment of 1-min
        
        """

        values_per_minute = int(fs*60)
        
        n_chunks = len(cleaned_sao2) // values_per_minute
        trimmed_sao2 = cleaned_sao2[:int(n_chunks*values_per_minute)]
        chunks = np.reshape(trimmed_sao2, (n_chunks, values_per_minute))

        #features = ["Smini", "Smean", "Svari", "CorreC", "NumZC", "Slope", "AbsSlope", "Bias", "Dmean", "ApExt", "LZCom", "CTM25", "CTM50", "CTM75", "CTM100", "odi2", "odi3", "odi4", "ODI21", "ODI23", "ODI25", "ODI31", "ODI33", "ODI35", "ODI41", "ODI43", "ODI45", "ODI51", "ODI53", "ODI55", "ODIS2", "ODIS3", "ODIS4", "ODIS5", "tsa70", "tsa80", "tsa85", "tsa90", "tsa95"]

        chunks_and_features = []
        
        for i, chunk in enumerate(chunks):
            
            if mask[i] == True:

                smini = np.min(chunk) if chunk.size > 0 else 0.0

                smean = np.mean(chunk) if chunk.size > 0 else 0.0

                svari = np.var(chunk)

                correc = np.corrcoef(np.arange(len(chunk)), chunk)[0, 1] 

                mask_means = chunk > smean # kann man einfach ignorieren, das hier Daten fehlen?
                #diff: difference between consecutive values (array) --> where: where result -1 (downward) or 1 (upward) (array indeces) --> length: number of events
                numzc = len(np.where(np.diff(mask_means.astype(float)) != 0)[0])

                x = np.arange(len(chunk))
                slope, bias = np.polyfit(x, chunk, 1) # intercept = bias
                absslope = np.abs(slope)

                dmean = np.mean(np.abs(np.diff(np.mean(chunk.reshape(5,12), axis=1))))

                apent = self.approximate_entropy(chunk)

                lzcom = self.lempel_ziv_complexity(chunk)

                ctm25 = self.central_tendency_measure(chunk, r= 0.0025) 
                ctm50 = self.central_tendency_measure(chunk, r=0.0050)
                ctm75 = self.central_tendency_measure(chunk, r=0.0075)
                ctm100 = self.central_tendency_measure(chunk, r=0.0100)

                odi2, odi3, odi4 = self.oxygen_desaturation_index(chunk)

                odi21 = self.odixy(chunk, x=2, y=1)
                odi23 = self.odixy(chunk, x=2, y=3)
                odi25 = self.odixy(chunk, x=2, y=5)
                odi31 = self.odixy(chunk, x=3, y=1)
                odi33 = self.odixy(chunk, x=3, y=3)
                odi35 = self.odixy(chunk, x=3, y=5)
                odi41 = self.odixy(chunk, x=4, y=1)
                odi43 = self.odixy(chunk, x=4, y=3)
                odi45 = self.odixy(chunk, x=4, y=5)
                odi51 = self.odixy(chunk, x=5, y=1)
                odi53 = self.odixy(chunk, x=5, y=3)
                odi55 = self.odixy(chunk, x=5, y=5)

                odis2, odis3, odis4, odis5 = self.odis(chunk)

                tsa70 = np.sum((chunk < 0.7).astype(int))
                tsa80 = np.sum((chunk < 0.8).astype(int))
                tsa85 = np.sum((chunk < 0.85).astype(int))
                tsa90 = np.sum((chunk < 0.9).astype(int))
                tsa95 = np.sum((chunk < 0.95).astype(int))

                new_row = {
                    "Smini": smini, "Smean": smean, "Svari": svari, "CorreC": correc, "NumZC": numzc, "Slope": slope, "AbsSlope": absslope, "Bias": bias, 
                    "Dmean": dmean, "ApEnt": apent, "LZCom": lzcom, "CTM25": ctm25, "CTM50": ctm50, "CTM75": ctm75, "CTM100": ctm100, 
                    "odi2": odi2, "odi3": odi3, "odi4": odi4, "ODI21": odi21, "ODI23": odi23, "ODI25": odi25, "ODI31": odi31, "ODI33": odi33, 
                    "ODI35": odi35, "ODI41": odi41, "ODI43": odi43, "ODI45": odi45, "ODI51": odi51, "ODI53": odi53, "ODI55": odi55,
                    "ODIS2": odis2, "ODIS3": odis3, "ODIS4": odis4, "ODIS5": odis5, "tsa70": tsa70, "tsa80": tsa80, "tsa85": tsa85, "tsa90": tsa90, "tsa95": tsa95
                }
            
            else:
                new_row = {
                    "Smini": None, "Smean": None, "Svari": None, "CorreC": None, "NumZC": None, "Slope": None, "AbsSlope": None, "Bias": None, 
                    "Dmean": None, "ApEnt": None, "LZCom": None, "CTM25": None, "CTM50": None, "CTM75": None, "CTM100": None, 
                    "odi2": None, "odi3": None, "odi4": None, "ODI21": None, "ODI23": None, "ODI25": None, "ODI31": None, "ODI33": None, 
                    "ODI35": None, "ODI41": None, "ODI43": None, "ODI45": None, "ODI51": None, "ODI53": None, "ODI55": None,
                    "ODIS2": None, "ODIS3": None, "ODIS4": None, "ODIS5": None, "tsa70": None, "tsa80": None, "tsa85": None, "tsa90": None, "tsa95": None
                }

            chunks_and_features.append(new_row)
        

        df = pd.DataFrame(chunks_and_features)
        #df.to_csv("C:/Users/timje/Documents/Studium/bachelor/code/feature_tables/spO2_features.csv", index=True)

        return df

    def get_features(self):
        """Return 2D pandas dataframe with SpO2 features per episode of 60s. First row contains feature names."""
        return self.features_per_episode
