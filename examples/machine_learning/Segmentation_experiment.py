#################################
from ayx import Alteryx
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from time import perf_counter
import gc

# from sklearn.compose import ColumnTransformer
# import numpy as np
# from sklearn.decomposition import PCA
# import pickle
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns


#################################
RUN_TRAIN = False  # turnoff to run alteryx quicker
MAX_CLUSTER = 15
MIN_CLUSTER = 2
BATCH_SIZE = 4196
# model = KMeans(random_state=2021, init='k-means++', max_iter=300)
model = MiniBatchKMeans(
    random_state=2021, init="k-means++", max_iter=300, batch_size=BATCH_SIZE
)
BASE_PATH = r"xxxx"
chart_size = (400, 200)
messages = []


#################################
# Read data
data = Alteryx.read("#1")
experiments = Alteryx.read("#2")

# save memory by reducing data down to only needed columns
features_all = experiments["Name"].values.tolist()
data = data[features_all]


# Get a list of the features that need to be scaled
features_scale = experiments[(experiments["Scale"] == 1)]
features_scale = features_scale["Name"].unique()

# Scale the features that need to be
print(f"Scale")
scaler = StandardScaler()
data[features_scale] = scaler.fit_transform(data[features_scale])

del scaler
gc.collect()


#################################
# extract a list of experiment labels
experiment_labels = experiments["Experiment"].unique()
experiment_number = len(experiment_labels)

# setup plots for combined experiments chart
fig, ax = plt.subplots(experiment_number, 3, figsize=(27, 5 * experiment_number))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# iterate through all the experiments and save the results to file
experiment_complete = 0
raw_data = []
messages.append(
    {"type": "Message", "message": f"Running {experiment_number} experiments"}
)
for i, label in enumerate(experiment_labels):
    try:
        if not RUN_TRAIN:
            raise UserWarning("Exit Early")

        messages.append({"type": "Message", "message": f"Running experiment {label}"})
        print(f"Running experiment {label}")
        # Filter to just the featues in this experiment
        features = experiments[experiments["Experiment"] == label]
        features = features["Name"].values.tolist()

        # Setup output path for this experiment and create folder if does not exist
        output_path = BASE_PATH + f"\\{label}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(f"{label} - Inertia")
        # Elbow using distortion to find optimal number of Cluster numbers
        timer_start = perf_counter()  # Start the stopwatch
        inertia = KElbowVisualizer(model, k=(MIN_CLUSTER, MAX_CLUSTER), ax=ax[i][0])
        inertia.fit(data[features])
        inertia.finalize()
        timer_stop = perf_counter()
        elasped_time = timer_stop - timer_start
        messages.append({"type": "Timer", "message": f"inertia:{elasped_time}"})

        print(f"{label} - Calinski Harabasz")
        # Elbow using calinski_harabasz to find optimal number of Cluster numbers
        timer_start = perf_counter()  # Start the stopwatch
        calinski_harabasz = KElbowVisualizer(
            model, k=(MIN_CLUSTER, MAX_CLUSTER), metric="calinski_harabasz", ax=ax[i][1]
        )
        calinski_harabasz.fit(data[features])
        calinski_harabasz.finalize()
        timer_stop = perf_counter()
        elasped_time = timer_stop - timer_start
        messages.append(
            {"type": "Timer", "message": f"calinski_harabasz:{elasped_time}"}
        )

        # Find intercluster distance for the elbow k value based of inertia chart
        print(f"{label} - PCA")
        clusters = inertia.elbow_value_
        # model_pca = KMeans(n_clusters=clusters, random_state=2021, init='k-means++', max_iter=300)
        model_pca = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=2021,
            init="k-means++",
            max_iter=300,
            batch_size=BATCH_SIZE,
        )

        timer_start = perf_counter()  # Start the stopwatch
        intercluster_distance = InterclusterDistance(model_pca, ax=ax[i][2])
        intercluster_distance.fit(data[features])
        intercluster_distance.finalize()
        timer_stop = perf_counter()
        elasped_time = timer_stop - timer_start
        messages.append(
            {"type": "Timer", "message": f"intercluster_distance:{elasped_time}"}
        )

        # Add Label at midpoint
        bbox_1 = (
            ax[i][0]
            .get_window_extent()
            .transformed(fig.transFigure.inverted())
            .get_points()
        )
        mid_point = (bbox_1[0][1] + bbox_1[1][1]) / 2
        fig.text(
            0.05,
            mid_point,
            label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=14,
        )

        # Save the indvidual experiment as it own image
        bbox_1 = (
            ax[i][0]
            .get_window_extent()
            .transformed(fig.dpi_scale_trans.inverted())
            .get_points()
        )
        bbox_2 = (
            ax[i][2]
            .get_window_extent()
            .transformed(fig.dpi_scale_trans.inverted())
            .get_points()
        )
        extent = Bbox([bbox_1[0], bbox_2[1]])
        fig.savefig(
            output_path + f"\\{label}_chart.png", bbox_inches=extent.expanded(1.2, 1.3)
        )

        # output the raw data to csv
        out_data = {
            "experiment": [label for k in range(MIN_CLUSTER, MAX_CLUSTER)],
            "k": [k for k in range(MIN_CLUSTER, MAX_CLUSTER)],
            "inertia_score": inertia.k_scores_,
            "inertia_timer": inertia.k_timers_,
            "inertia_elbow": [
                inertia.elbow_value_ for i in range(MIN_CLUSTER, MAX_CLUSTER)
            ],
            "calinski_harabasz_score": calinski_harabasz.k_scores_,
            "calinski_harabasz_timer": calinski_harabasz.k_timers_,
            "calinski_harabasz_elbow": [
                calinski_harabasz.elbow_value_ for i in range(MIN_CLUSTER, MAX_CLUSTER)
            ],
        }

        raw_data.append(out_data)
        out_data = pd.DataFrame(out_data)
        out_data.to_csv(output_path + f"\\{label}_data.csv", index=False)
        experiment_complete += 1
    except Exception as err:
        print(err)
        messages.append({"type": "Error", "message": err})


#################################
messages.append(
    {
        "type": "Message",
        "message": f"{experiment_complete} of {experiment_number} experiments completed",
    }
)
fig.savefig(BASE_PATH + "\\cluster_eval_overall.png")

if raw_data:
    out_data = pd.concat(pd.DataFrame(x) for x in raw_data)
    out_data.to_csv(BASE_PATH + f"\\data.csv", index=False)
    Alteryx.write(out_data, 1)

messages_df = pd.DataFrame(messages)
messages_df.to_csv(BASE_PATH + f"\\messages.csv", index=False)
Alteryx.write(messages_df, 1)


#################################
# #extract a list of experiment labels
# experiment_labels = experiments['Experiment'].unique()
# clusters = [5,7]
# experiment_number = len(clusters)
# label = [x for x in experiment_labels if x == 'exp_04'][0]

# #setup plots for combined experiments chart
# fig_pca, ax_pca = plt.subplots(experiment_number,1,figsize=(9, 7*experiment_number))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)


# #Filter to just the featues in this experiment
# features = experiments[experiments['Experiment']==label]
# features = features['Name'].values.tolist()
# #Get a list of the features that need to be scaled
# features_scale = experiments[(experiments['Experiment']==label) & (experiments['Scale'] ==1)]
# features_scale = features_scale['Name'].values.tolist()

# #Setup output path for this experiment and create folder if does not exist
# output_path  = BASE_PATH + f'\\{label}'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# #Scale the features that need to be
# scaler = StandardScaler()
# data_scaled = data[features].copy()
# data_scaled[features] = scaler.fit_transform(data_scaled[features])


# for i, cluster in enumerate(clusters):
#         model_pca = MiniBatchKMeans(n_clusters=cluster, random_state=2021, init='k-means++', max_iter=300, batch_size=8192)
#         timer_start = perf_counter() # Start the stopwatch
#         intercluster_distance = InterclusterDistance(model_pca,ax=ax_pca[i])
#         intercluster_distance.fit(data_scaled)
#         intercluster_distance.finalize()
#         timer_stop = perf_counter()
#         elasped_time = timer_stop - timer_start
#         messages.append({'type':'Timer', 'message': f'intercluster_distance:{elasped_time}'})

#         #add label to the chart
#         bbox_1 = ax_pca[i].get_window_extent().transformed(fig_pca.transFigure.inverted()).get_points()
#         mid_point = (bbox_1[0][1] + bbox_1[1][1])/2
#         fig_pca.text(0.05, mid_point, cluster, ha='center', va='center', rotation='vertical', fontsize=14)

# fig_pca.savefig(BASE_PATH +f'\\pca_{label}.png')


#################################
