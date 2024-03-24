from scipy.io import loadmat
import numpy as np


class TimeSeriesProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeSeriesProcessor, cls).__new__(cls)
        return cls._instance

    def process_and_save(self, filenames, output_file):
        all_averaged_samples = []

        for filename in filenames:
            data = loadmat(filename)
            keys = list(data.keys())
            eeg_samples = data[keys[3]]

            averaged_samples = np.mean(eeg_samples, axis=0)

            print(averaged_samples[:10])
            num_chunks = len(averaged_samples) // 14000
            print(len(averaged_samples))
            trimmed_samples = averaged_samples[:(num_chunks*14000)]

            print(trimmed_samples[:10])
            all_averaged_samples.append(trimmed_samples)

        concatenated_data = np.concatenate(all_averaged_samples)

        print(concatenated_data[:10])
        np.savetxt(output_file, concatenated_data)


if __name__ == "__main__":
    filenames = ['1pilot7333_20231213_013546.mat', '2pilot0718_20231213_081206.mat',
                 '3pilot6411_20240123_010146.mat', '4pilot8314_20240124_063626.mat',
                 '5pilot9222_20240126_125518.mat', '6pilot1515_20240126_040300.mat',
                 '7pilot7477_20240127_035541.mat', '8pilot1632_20240127_063737.mat']

    output_file = 'processed_data.txt'

    processor = TimeSeriesProcessor()
    processor.process_and_save(filenames, output_file)
