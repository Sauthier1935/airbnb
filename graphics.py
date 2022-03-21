import seaborn as sns
import matplotlib.pyplot as plt
import utils
import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd

class Graphics_utils:

    def plot_heatmap(df):
        plt.figure(figsize=(25, 25))
        sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
        plt.show()

    def box_diagram(col):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        sns.boxplot(x=col, ax=ax1)
        ax2.set_xlim(utils.Utils.limit(col))
        sns.boxplot(x=col, ax=ax2)
        plt.show()

    def histogram(col):
        plt.figure(figsize=(15, 5))
        sns.distplot(col, hist=True)
        plt.show()

    def barr_graph(col):
        plt.figure(figsize=(15, 5))
        ax = sns.barplot(x=col.value_counts().index, y=col.value_counts())
        ax.set_xlim(utils.Utils.limit(col))
        plt.show()

    def barr_target(df):
        plt.figure(figsize=(15, 5))
        sns.displot(df)
        plt.show()

    def text_attribute_graph(df,col):
        plt.figure(figsize=(15, 5))
        graph = sns.countplot(col, data=df)
        graph.tick_params(axis='x', rotation=90)
        plt.show()

    def plot_map(df):
        sample = df.sample(n=50000)
        center_map = {'lat': sample.latitude.mean(), 'lon': sample.longitude.mean()}
        rj_map = px.density_mapbox(sample, lat='latitude', lon='longitude', z='price', radius=2.5,
                                   center=center_map, zoom=10,
                                   mapbox_style='stamen-terrain')
        rj_map.show()

    def plot_confusion_matrix(cf,lbl1,lbl2):
        plt.figure(figsize=(15, 10))
        sns.heatmap(cf, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
        plt.show()

    def build_clusters(target):
        num_of_clusters = range(2, 10)
        error = []
        for num_clusters in num_of_clusters:
            clusters = KMeans(num_clusters)
            clusters.fit(target)
            error.append(clusters.inertia_ / 100)
        df_test = pd.DataFrame({"Cluster_Numbers": num_of_clusters, "Error_Term": error})

        plt.figure(figsize=(15, 10))
        plt.plot(df_test.Cluster_Numbers, df_test.Error_Term, marker="D", color='red')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('Find the optimal number of cluster')
        plt.show()

    def plot_clusters_k_means(x_train_arr,y_kmeans,kmeans):
        # Visualising the clusters and Plotting the centroids of the clusters
        plt.scatter(x_train_arr[y_kmeans == 0, 0], x_train_arr[y_kmeans == 0, 0], s=100, c='red', label='0')
        plt.scatter(x_train_arr[y_kmeans == 1, 0], x_train_arr[y_kmeans == 1, 0], s=100, c='blue', label='1')
        plt.scatter(x_train_arr[y_kmeans == 2, 0], x_train_arr[y_kmeans == 2, 0], s=100, c='green', label='2')
        plt.scatter(x_train_arr[y_kmeans == 3, 0], x_train_arr[y_kmeans == 3, 0], s=100, c='orange', label='3')
        # Plotting the centroids of the clusters
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
        plt.xlabel("Price")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
