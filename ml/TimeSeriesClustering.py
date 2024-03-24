import numpy as np
from sklearn.cluster import KMeans
from parameters.LargestLyapunovParameter import LargestLyapunovExponent
from parameters.Entropy import Entropy
from parameters.Dimension import BoxCountingDimension
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class TimeSeriesClustering:
    def __init__(self, filename, chunk_size=14000):
        self.filename = filename
        self.chunk_size = chunk_size
        self.data = None

    def read_data(self):
        try:
            with open(self.filename, 'r') as file:
                self.data = [float(line.strip()) for line in file]
        except FileNotFoundError:
            print(f"Файл {self.filename} не найден.")
            return False
        return True

    def process_time_series(self):
        if self.data is None:
            print("Данные не загружены.")
            return

        time_series_list = []
        num_chunks = len(self.data) // self.chunk_size

        for i in range(num_chunks):
            chunk_data = self.data[i * self.chunk_size:(i + 1) * self.chunk_size]

            parameters = self.calculate_parameters(chunk_data)
            time_series_list.append(parameters)

        self.cluster_time_series(time_series_list)

    def calculate_parameters(self, time_series_data):
        mean_value = np.mean(time_series_data)
        std_deviation = np.std(time_series_data)
        max_value = np.max(time_series_data)
        min_value = np.min(time_series_data)

        dimension_calculating = BoxCountingDimension()
        dimension = dimension_calculating.calculate(time_series_data)

        entropy_calculating = Entropy()
        entropy = entropy_calculating.calculate(time_series_data)

        time_delay = 1
        lyapunov_calculator = LargestLyapunovExponent(dimension // 1, time_delay)
        lyapunov_exponent = lyapunov_calculator.calculate(time_series_data)
        return [mean_value, std_deviation, max_value, min_value, dimension, entropy, lyapunov_exponent]

    def cluster_time_series(self, time_series_list):
        if not time_series_list:
            print("Список временных рядов пуст.")
            return

        k = 3
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(time_series_list)

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(time_series_list)
        principal_components = pca.components_

        print("Коэффициенты (веса) для каждой главной компоненты:")
        for i, component in enumerate(principal_components):
            print(f"Главная компонента {i + 1}:")
            for j, parameter_name in enumerate(['Энтропия', 'Старший показатель Ляпунова', 'Размерность']):
                coefficient = component[j]
                print(f"  Коэффициент для {parameter_name}: {coefficient:.3f}")

        plt.figure(figsize=(8, 6))
        for i in range(k):
            cluster_members = np.where(clusters == i)[0]
            plt.scatter(reduced_features[cluster_members, 0], reduced_features[cluster_members, 1],
                        label=f'Кластер {i + 1}')

        plt.title('Визуализация кластеров с использованием PCA')
        plt.xlabel('Главная компонента 1')
        plt.ylabel('Главная компонента 2')
        plt.legend()
        plt.show()

        print("Результаты кластеризации:")
        for i in range(k):
            cluster_members = np.where(clusters == i)[0]
            print(f"Кластер {i + 1}: {len(cluster_members)} временных ряда(-ов)")


if __name__ == "__main__":
    filename = "processed_data.txt"
    ts_clustering = TimeSeriesClustering(filename)
    if ts_clustering.read_data():
        ts_clustering.process_time_series()
