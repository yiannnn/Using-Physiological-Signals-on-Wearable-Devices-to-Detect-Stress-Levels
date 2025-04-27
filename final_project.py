#!/usr/bin/env python
# coding: utf-8

# # Readiness - Cheng-Huan Yu

# In[35]:


import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy as np
import plotly.express as px
import scipy.stats as stats


# In[8]:


# base_dir = "/Users/andrewyu/Desktop/Git/SJSU/Thesis/pmdata/"
# well=pd.read_csv('/Users/andrewyu/Desktop/Git/SJSU/Thesis/pmdata/p01/pmsys/wellness.csv')

base_dir = "pmdata/"
well = pd.read_csv('pmdata/p01/pmsys/wellness.csv')

def combine_all(base_dir):
    participants = [f"p{str(i).zfill(2)}" for i in range(1, 17) if i not in [12, 13]]

    fitbit_files = [
        "calories.json", "distance.json", "exercise.json",
        "heart_rate.json", "resting_heart_rate.json",
        "steps.json", "sleep.json", "sleep_score.csv"
    ]
    fitbit_files = os.listdir('pmdata/p01/fitbit')
    data_dict = {file: [] for file in fitbit_files}

    for participant in tqdm(participants):
        fitbit_path = os.path.join(base_dir, participant, "fitbit")

        if os.path.exists(fitbit_path):  
            for file in tqdm(fitbit_files):
                file_path = os.path.join(fitbit_path, file)

                if os.path.exists(file_path): 
                    if file.endswith(".json"):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        
                        df = pd.json_normalize(data)
                        df["participant_id"] = participant  
                        data_dict[file].append(df)

                    elif file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        df["participant_id"] = participant  
                        data_dict[file].append(df)
                else:
                    print(f'{file_path} no exist')
    return data_dict

def extract_time_domain_features(signal_data):
    features = {
        "mean": np.mean(signal_data, axis=0),
        "std": np.std(signal_data, axis=0),
        "max": np.max(signal_data, axis=0),
        "min": np.min(signal_data, axis=0),
        "p2p": np.ptp(signal_data, axis=0),  
        "rms": np.sqrt(np.mean(signal_data ** 2, axis=0)),  
        "skewness": stats.skew(signal_data, axis=0),
        "kurtosis": stats.kurtosis(signal_data, axis=0),
        "cv":        np.std(signal_data, axis=0) / (np.mean(signal_data, axis=0) + 1e-8),
        "iqr":       stats.iqr(signal_data, axis=0),
        "median":    np.median(signal_data, axis=0)
    }
    return pd.DataFrame(features)

def check_time_format(data_dict,well,p,mode=1,day=7,combined_features='all'):
    day-=1
    step1=data_dict['steps.json'][p-1]
    calories1=data_dict['calories.json'][p-1]
    resting_heart_rate1=data_dict['resting_heart_rate.json'][p-1]
    sleep1=data_dict['sleep.json'][p-1].loc[:,['dateOfSleep','minutesAsleep']]
    mood1=well.loc[:,['effective_time_frame','mood']]
    stress1=well.loc[:,['effective_time_frame','stress']]
    readiness1=well.loc[:,['effective_time_frame','readiness']]

    step1["dateTime"] = pd.to_datetime(step1["dateTime"]).dt.date
    step1['value']=step1['value'].astype('int')

    calories1["dateTime"] = pd.to_datetime(calories1["dateTime"]).dt.date
    calories1['value']=calories1['value'].astype('float')

    resting_heart_rate1["dateTime"] = pd.to_datetime(resting_heart_rate1["dateTime"]).dt.date
    sleep1["dateTime"] = pd.to_datetime(sleep1["dateOfSleep"]).dt.date

    mood1["dateTime"] = pd.to_datetime(mood1["effective_time_frame"]).dt.date
    stress1["dateTime"] = pd.to_datetime(stress1["effective_time_frame"]).dt.date

    readiness1["dateTime"] = pd.to_datetime(readiness1["effective_time_frame"]).dt.date

    if mode == 'stat':
        all_prompt=[]
        GT_mean=[]
        GT_median=[]
        dates=[]
        all_days_df = pd.DataFrame()

        for date in tqdm(sleep1['dateTime']):
            #try:
                
                mask = (step1['dateTime'] >= date - pd.Timedelta(days=day)) & (step1['dateTime'] <= date)
                step_sub = step1.loc[mask, ['value']]

                mask = (calories1['dateTime'] >= date - pd.Timedelta(days=day)) & (calories1['dateTime'] <= date)
                calories_sub=calories1.loc[mask,['value']]

                mask = (resting_heart_rate1['dateTime'] >= date - pd.Timedelta(days=day)) & (resting_heart_rate1['dateTime'] <= date)
                resting_heart_rate_sub=resting_heart_rate1.loc[mask,['value.value']]

                mask = (sleep1['dateTime'] >= date - pd.Timedelta(days=day)) & (sleep1['dateTime'] <= date)
                sleep_sub=sleep1.loc[mask,['minutesAsleep']]

                mask = (mood1['dateTime'] >= date - pd.Timedelta(days=day)) & (mood1['dateTime'] <= date)
                mood_sub=mood1.loc[mask,['mood']]


                mask = (stress1['dateTime'] >= date - pd.Timedelta(days=day)) & (stress1['dateTime'] <= date)
                stress_sub=stress1.loc[mask,['stress']]

                mask = (readiness1['dateTime'] >= date - pd.Timedelta(days=day)) & (readiness1['dateTime'] <= date)
                readiness_sub=readiness1.loc[mask,['readiness']]



                step_sub_stat=extract_time_domain_features(step_sub)
                calories_sub_stat=extract_time_domain_features(calories_sub)
                resting_heart_rate_sub_stat=extract_time_domain_features(resting_heart_rate_sub)
                mood_sub_stat=extract_time_domain_features(mood_sub)
                sleep_sub_stat=extract_time_domain_features(sleep_sub)
                readiness_substat=extract_time_domain_features(readiness_sub)
            

                data_out = {
                'Steps': step_sub_stat.loc['value'].to_dict(),
                'Calories Burn':calories_sub_stat.loc['value'].to_dict(),
                'Resting Heart Rate': resting_heart_rate_sub_stat.loc['value.value'].to_dict(),
                'Sleep Duration': sleep_sub_stat.loc['minutesAsleep'].to_dict(),
                'Mood': mood_sub_stat.loc['mood'].to_dict()
                }

                data_out = pd.DataFrame(data_out) 
                
                flat_df = data_out.T.reset_index().melt(id_vars='index')
                flat_df['feature'] = flat_df['index'] + '_' + flat_df['variable']
                flat_result = flat_df[['feature', 'value']].set_index('feature').T
                flat_result.insert(0, 'date', date)
                flat_result.insert(len(flat_result), 'GT_mean', readiness_sub.mean()['readiness'])
                flat_result.insert(len(flat_result), 'GT_median', readiness_sub.median()['readiness'])

                
                all_days_df = pd.concat([all_days_df, flat_result], ignore_index=True)

        return 1,all_days_df


# In[9]:


data_dict=combine_all(base_dir)


train_list = []
test_list = []

for i in range(1, 15):
    _, df = check_time_format(data_dict, well, i, 'stat', day=14)
    if i <= 10:
        train_list.append(df)
    else:
        test_list.append(df)

train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)


train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))  # 或 train_df 的 median


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lazypredict.Supervised import LazyClassifier

from lazypredict.Supervised import LazyRegressor


# In[11]:


type='GT_median'

X = train_df.drop(columns=['date', 'GT_median', 'GT_mean'])
y = train_df[type] 

test_X = test_df.drop(columns=['date', 'GT_median', 'GT_mean'])
test_y = test_df[type]  


reg = LazyRegressor(
    verbose=0,
    ignore_warnings=True,
    custom_metric=mean_absolute_error 

)

models, predictions = reg.fit(X, test_X,y, test_y)


print(models)


# # Sleep Quality - Chun-Chieh Kuo

# In[18]:


import os
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


# # 1. Loading data

# In[12]:


base_dir = "pmdata/"
well = pd.read_csv('pmdata/p01/pmsys/wellness.csv')


def combine_all(base_dir):
    participants = [f"p{str(i).zfill(2)}" for i in range(1, 17)]
    fitbit_files = ["sleep.json", "sleep_score.csv"]
    data_dict = {file: [] for file in fitbit_files}

    for participant in tqdm(participants):
        fitbit_path = os.path.join(base_dir, participant, "fitbit")

        if os.path.exists(fitbit_path):
            for file in fitbit_files:
                file_path = os.path.join(fitbit_path, file)

                if os.path.exists(file_path):
                    try:
                        if file.endswith(".json"):
                            with open(file_path, "r") as f:
                                data = json.load(f)
                            df = pd.json_normalize(data)
                            df["participant_id"] = participant
                            data_dict[file].append(df)
                        elif file.endswith(".csv"):
                            df = pd.read_csv(file_path)
                            df["participant_id"] = participant
                            data_dict[file].append(df)
                    except Exception as e:
                        print(f"Error reading {file} from {participant}: {e}")
                else:
                    print(f"Missing {file} in {participant}")

    return data_dict


def check_time_format(data_dict, well, p, mode=1, day=7):
    sleep_score = data_dict['sleep_score.csv'][p - 1]
    sleep_json = data_dict['sleep.json'][p - 1]

    # Extract relevant columns from wellness.csv
    mood = well.loc[:, ['effective_time_frame', 'mood']]
    stress = well.loc[:, ['effective_time_frame', 'stress']]
    fatigue = well.loc[:, ['effective_time_frame', 'fatigue']]
    readiness = well.loc[:, ['effective_time_frame', 'readiness']]
    sleep_quality = well.loc[:, ['effective_time_frame', 'sleep_quality']]

    # Convert date formats
    sleep_score["dateTime"] = pd.to_datetime(sleep_score.iloc[:, 0]).dt.date
    for df in [mood, stress, fatigue, readiness, sleep_quality]:
        df["dateTime"] = pd.to_datetime(df["effective_time_frame"]).dt.date

    # Preprocess sleep.json
    sleep_json["dateTime"] = pd.to_datetime(sleep_json["dateOfSleep"]).dt.date
    sleep_json_flat = sleep_json.loc[:, ['dateTime', 'efficiency', 'minutesToFallAsleep']]

    all_data = []

    for date in tqdm(sleep_score["dateTime"]):
        try:
            row = {"dateTime": date}

            # Extract scores and resting heart rate
            row.update(sleep_score[sleep_score["dateTime"] == date][[
                'overall_score', 'composition_score', 'revitalization_score',
                'duration_score', 'deep_sleep_in_minutes', 'restlessness',
                'resting_heart_rate'
            ]].mean(numeric_only=True).to_dict())

            # Add sleep efficiency and fall asleep time
            row.update(sleep_json_flat[sleep_json_flat["dateTime"] == date][[
                "efficiency", "minutesToFallAsleep"
            ]].mean(numeric_only=True).to_dict())

            # Add wellness metrics
            row["mood"] = mood[mood["dateTime"] == date]["mood"].mean()
            row["stress"] = stress[stress["dateTime"] == date]["stress"].mean()
            row["fatigue"] = fatigue[fatigue["dateTime"] == date]["fatigue"].mean()
            row["readiness"] = readiness[readiness["dateTime"] == date]["readiness"].mean()
            row["sleep_quality"] = sleep_quality[sleep_quality["dateTime"] == date]["sleep_quality"].mean()

            all_data.append(row)
        except Exception as e:
            print(f"Missing or invalid data on {date}: {e}")

    return pd.DataFrame(all_data)


# In[13]:


data_dict = combine_all(base_dir)

all_participant_data = []

for p in range(1, 17):
    df = check_time_format(data_dict, well, p)
    df["participant_id"] = f"p{str(p).zfill(2)}"
    all_participant_data.append(df)
     
all_sleep_df = pd.concat(all_participant_data, ignore_index=True)

all_sleep_df


# # 2. Feature Engineering

# In[14]:


# handle missing values
cols_to_fill = ['mood', 'stress', 'fatigue', 'readiness', 'sleep_quality']
for col in cols_to_fill:
    all_sleep_df[col] = all_sleep_df.groupby("participant_id")[col].transform(lambda x: x.fillna(x.median()))

    
for col in ['efficiency', 'minutesToFallAsleep']:
    median_val = all_sleep_df[col].median()
    all_sleep_df[col] = all_sleep_df[col].fillna(median_val)


# In[16]:


all_sleep_df


# In[7]:


sns.histplot(data=all_sleep_df, x="overall_score", kde=True)
plt.title("Distribution of Fitbit Sleep Scores")
plt.show()


# In[8]:


sns.boxplot(data=all_sleep_df, x="participant_id", y="overall_score")
plt.title("Sleep Score by Participant")
plt.xticks(rotation=45)
plt.show()


# In[9]:


sns.boxplot(data=all_sleep_df, x="sleep_quality", y="overall_score")
plt.title("Minutes Asleep by Sleep Quality Label")
plt.show()


# In[10]:


plt.figure(figsize=(12, 10))
sns.heatmap(all_sleep_df.select_dtypes(include='number').corr(), annot=False, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[11]:


participants = sorted(all_sleep_df["participant_id"].unique())
fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(16, 24), sharex=False, sharey=False)
axes = axes.flatten()  

for i, user in enumerate(participants):
    ax = axes[i]
    df_user = all_sleep_df[all_sleep_df["participant_id"] == user]
    ax.plot(df_user["dateTime"], df_user["overall_score"], marker='o')
    ax.set_title(f"{user} – Sleep Score", fontsize=10)
    ax.tick_params(axis='x', rotation=45)

for j in range(len(participants), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# # 3. Dimensionality Reduction

# In[19]:


# Variance Threshold
df_numeric = all_sleep_df.select_dtypes(include='number')

selector = VarianceThreshold(threshold=0.01)
reduced_df = selector.fit_transform(df_numeric)

retained_columns = df_numeric.columns[selector.get_support()]
df_reduced_var = pd.DataFrame(reduced_df, columns=retained_columns)

print("Remaining columns:", list(retained_columns))


# In[20]:


df_reduced_var


# In[21]:


# Correlation Filter
corr_matrix = df_reduced_var.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

df_reduced_corr = df_reduced_var.drop(columns=to_drop)

print("Dropped:", to_drop)
print("Remaining columns:", df_reduced_corr.columns.tolist())


# In[22]:


df_reduced_corr


# In[23]:


# Lasso
X = df_reduced_corr.drop(columns=['overall_score']) 
y = df_reduced_corr['overall_score']               

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.05) 
lasso.fit(X_scaled, y)

coef = pd.Series(lasso.coef_, index=X.columns)
selected_features = coef[coef != 0].index.tolist()

print("Lasso selected features:", selected_features)

df_lasso_selected = df_reduced_corr[selected_features + ['overall_score']]


# In[24]:


df_lasso_selected


# # 4. Regression

# In[26]:


X = df_lasso_selected.drop(columns=["overall_score"])
y = df_lasso_selected["overall_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR (RBF Kernel)": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)),
    "KNN Regression": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)) 
}


results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "MSE": round(mse, 4)
    })

results_df = pd.DataFrame(results)
results_df


# In[29]:


X = df_lasso_selected.drop(columns=["overall_score"])
y = df_lasso_selected["overall_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

models


# # 5. Calssification

# In[28]:


df_lasso_selected["sleep_good"] = (df_lasso_selected["sleep_quality"] >= 3).astype(int)
X = df_lasso_selected.drop(columns=["sleep_quality", "overall_score", "sleep_good"])
y = df_lasso_selected["sleep_good"]

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    accs, f1s, precs, recs = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        recs.append(recall_score(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": round(np.mean(accs), 4),
        "F1 Score": round(np.mean(f1s), 4),
        "Precision": round(np.mean(precs), 4),
        "Recall": round(np.mean(recs), 4)
    })

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
results_df


# In[30]:


df_lasso_selected["sleep_good"] = (df_lasso_selected["sleep_quality"] >= 3).astype(int)
X = df_lasso_selected.drop(columns=["sleep_quality", "overall_score", "sleep_good"])
y = df_lasso_selected["sleep_good"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

models


# # 6. Clustering

# In[20]:


cluster_features = [
    'composition_score', 'revitalization_score', 'deep_sleep_in_minutes',
    'resting_heart_rate', 'efficiency', 'minutesToFallAsleep',
    'mood', 'stress'
]

X_cluster = df_lasso_selected[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.grid(True)
plt.show()


# In[21]:


# K-Means
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df_lasso_selected.copy()
df_clustered["cluster"] = clusters

cluster_summary = df_clustered.groupby("cluster")[cluster_features].mean().round(2)
display(cluster_summary)


# In[22]:


from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["cluster"] = clusters

plt.figure(figsize=(6, 5))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="Set2", s=40)
plt.title(f"K-Means Clustering (k = {best_k}) - PCA Projection")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[23]:


kmeans_summary = df_clustered.groupby("cluster")[cluster_features].mean().reset_index()

kmeans_norm = kmeans_summary.copy()
kmeans_norm[cluster_features] = (kmeans_norm[cluster_features] - kmeans_norm[cluster_features].min()) /                                 (kmeans_norm[cluster_features].max() - kmeans_norm[cluster_features].min())

labels = cluster_features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] 

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(len(kmeans_norm)):
    values = kmeans_norm.loc[i, cluster_features].tolist()
    values += values[:1]  # loop close
    ax.plot(angles, values, label=f"Cluster {int(kmeans_norm.loc[i, 'cluster'])}")
    ax.fill(angles, values, alpha=0.1)

ax.set_title("K-Means Clustering - Radar Chart by Group", y=1.08)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_rlabel_position(0)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()


# In[24]:


# Hierarchical dendrogram
X_sample = X_scaled[:50]
linked = linkage(X_sample, method='ward')


plt.figure(figsize=(18, 15))
dendrogram(linked, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram (First 50 rows)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# In[25]:


# # Hierarchical Clustering
linked_all = linkage(X_scaled, method='ward')

k_hc = 3
hc_labels = fcluster(linked_all, t=k_hc, criterion='maxclust')

df_hc = df_lasso_selected.copy()
df_hc["hc_cluster"] = hc_labels

hc_summary = df_hc.groupby("hc_cluster")[cluster_features].mean().round(2)
display(hc_summary)


# In[26]:


from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["hc_cluster"] = hc_labels  

plt.figure(figsize=(6, 5))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="hc_cluster", palette="Set2", s=40)
plt.title(f"Hierarchical Clustering (k = {k_hc}) - PCA Projection")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


cluster_avg = df_hc.groupby("hc_cluster")[cluster_features].mean()
cluster_avg = cluster_avg.reset_index()

cluster_avg_norm = cluster_avg.copy()
cluster_avg_norm[cluster_features] = (cluster_avg_norm[cluster_features] - cluster_avg_norm[cluster_features].min()) /                                      (cluster_avg_norm[cluster_features].max() - cluster_avg_norm[cluster_features].min())

labels = cluster_features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(len(cluster_avg_norm)):
    values = cluster_avg_norm.loc[i, cluster_features].tolist()
    values += values[:1]  # close the loop
    ax.plot(angles, values, label=f"Cluster {int(cluster_avg_norm.loc[i, 'hc_cluster'])}")
    ax.fill(angles, values, alpha=0.1)

ax.set_title("Hierarchical Clustering - Radar Chart by Group", y=1.08)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_rlabel_position(0)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()


# # Stress - Shao-Yu Huang

# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
import smogn


# In[7]:


base_dir = "pmdata"
participants = [f"p{str(i).zfill(2)}" for i in range(1, 17)]

def combine_all(base_dir):
    fitbit_files = os.listdir(os.path.join(base_dir, "p01", "fitbit"))
    data_dict = {file: {} for file in fitbit_files}

    for participant in tqdm(participants):
        fitbit_path = os.path.join(base_dir, participant, "fitbit")
        if os.path.exists(fitbit_path):
            for file in fitbit_files:
                file_path = os.path.join(fitbit_path, file)
                if os.path.exists(file_path):
                    if file.endswith(".json"):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        df = pd.json_normalize(data)
                    elif file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                    df["participant_id"] = participant
                    data_dict[file][participant] = df
    return data_dict

def participant_has_data(data_dict, pid):
    required_keys = ['steps.json', 'calories.json', 'sleep.json']
    for key in required_keys:
        if pid not in data_dict[key]:
            print(f"Missing {key} for {pid}")
            return False
    return True

def check_time_format(data_dict, well, p):
    pid = f"p{i:02d}"
    step1 = data_dict['steps.json'].get(pid)
    calories1 = data_dict['calories.json'].get(pid)
    resting_heart_rate1 = data_dict['resting_heart_rate.json'].get(pid)
    sleep1_raw = data_dict['sleep.json'].get(pid)
    if sleep1_raw is None:
        print(f"{pid} missing sleep.json — skipping")
        return pd.DataFrame()  
    else:
        sleep1 = sleep1_raw.loc[:, ['dateOfSleep', 'minutesAsleep']]
    mood1 = well.loc[:, ['effective_time_frame', 'mood']]
    stress1 = well.loc[:, ['effective_time_frame', 'stress']]
    fatigue1 = well.loc[:, ['effective_time_frame', 'fatigue']]

    if resting_heart_rate1 is None:
        print(f"{pid} has no resting_heart_rate.json — HR will be NaN")

    step1["dateTime"] = pd.to_datetime(step1["dateTime"]).dt.date
    calories1["dateTime"] = pd.to_datetime(calories1["dateTime"]).dt.date
    if resting_heart_rate1 is not None:
        resting_heart_rate1["dateTime"] = pd.to_datetime(resting_heart_rate1["dateTime"]).dt.date
    sleep1["dateTime"] = pd.to_datetime(sleep1["dateOfSleep"]).dt.date
    mood1["dateTime"] = pd.to_datetime(mood1["effective_time_frame"]).dt.date
    stress1["dateTime"] = pd.to_datetime(stress1["effective_time_frame"]).dt.date
    fatigue1["dateTime"] = pd.to_datetime(fatigue1["effective_time_frame"]).dt.date

    step1['value'] = step1['value'].astype('int')
    calories1['value'] = calories1['value'].astype('float')

    all = []
    for date in tqdm(sleep1['dateTime']):
        try:
            l1 = [date]
            l1.append(np.sum(step1.loc[step1['dateTime'] == date, 'value']))
            l1.append(np.sum(calories1.loc[calories1['dateTime'] == date, 'value']))
            if resting_heart_rate1 is not None:
                hr = np.mean(resting_heart_rate1.loc[resting_heart_rate1['dateTime'] == date, 'value.value'])
            else:
                hr = np.nan  
            l1.append(hr)
            l1.append(np.mean(sleep1.loc[sleep1['dateTime'] == date, 'minutesAsleep']))
            l1.append(np.mean(mood1.loc[mood1['dateTime'] == date, 'mood']))
            l1.append(np.mean(stress1.loc[stress1['dateTime'] == date, 'stress']))
            l1.append(np.mean(fatigue1.loc[fatigue1['dateTime'] == date, 'fatigue']))
            all.append(l1)
        except Exception as e:
            print(e)
            print('Missing data on ' + str(date))

    columns = ['dateTime', 'step', 'calories']
    if resting_heart_rate1 is  None:
        print(f"{pid} has no resting_heart_rate.json — column skipped")
    columns.append('resting_heart_rate')
    columns += ['sleep', 'mood', 'stress', 'fatigue']
    
    return pd.DataFrame(all, columns=columns)

data_dict = combine_all(base_dir)

for i in range(1, 17):
    pid = f"p{i:02d}"
    well_path = os.path.join(base_dir, pid, "pmsys", "wellness.csv")
    if not os.path.exists(well_path):
        print(f"{pid} skipped: wellness.csv not found.")
        continue

    if not participant_has_data(data_dict, pid):
        print(f"{pid} skipped: missing Fitbit data.")
        continue

    try:
        well = pd.read_csv(well_path)
        df = check_time_format(data_dict, well, i)
        df.to_csv(f"{pid}_stat_data.csv", index=False)
        print(f"{pid}.csv done!")
    except Exception as e:
        print(f"{pid} error: {e}")


# In[8]:


import glob

# Combining all the files into one
all_files = glob.glob("p??_stat_data.csv")

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df["participant_id"] = file[:3]
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv("all_participants.csv", index=False)
print("all_participants.csv saved!")


# # Preprocessing

# In[9]:


df.dropna(inplace=True)
df = df.drop(columns=["dateTime", "participant_id"])
df.head()


# In[10]:


sns.histplot(df['stress'], bins=30, kde=True)
plt.title("Stress Distribution")
plt.xlabel("Stress Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[11]:


X = df.drop(columns=["stress"])
y = df["stress"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # PCA
# 

# In[12]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_pca[:, 1] *= -1

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title("PCA: First 2 Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[15]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.show()

clusters = fcluster(linked, t=3, criterion='maxclust')

X_pca[:, 1] *= -1
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set1', alpha=0.7)
plt.title("PCA with Hierarchical Clustering")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()


# # Lazy Prediction

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=1, ignore_warnings=True, predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)


# observation: the result is not that good!

# In[17]:


df["stress_class"] = df["stress"].astype(int)
df.head()


# # RandomForestClassifier
# 

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

X = df[['step', 'calories', 'resting_heart_rate', 'sleep', 'mood', 'fatigue']]
y = df["stress_class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # KNeighborsClassifier

# In[19]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[20]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np


# In[22]:


# 2. ROC Curve & AUC (binary class only)
# Binarize for ROC (assuming class 2 vs 3, so we simulate that)
y_bin = label_binarize(y, classes=[2, 3])
if y_bin.shape[1] == 1:  # Only one class in y -> no ROC possible
    roc_possible = False
else:
    roc_possible = True
    y_test_bin = label_binarize(y_test, classes=[2, 3])
    y_score = clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# In[23]:


# 3. Feature Importance
feature_cols = ['step', 'calories', 'resting_heart_rate', 'sleep', 'mood', 'fatigue']

importances = clf.feature_importances_
feature_names = feature_cols
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()


# In[29]:


# 4. t-SNE Projection
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)


X_tsne[:, 1] *= -1
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="Set1")
plt.title("t-SNE Projection of Stress Data")
plt.tight_layout()
plt.show()


# In[30]:


from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# For binary classification, relabel stress_class: class 2 -> 0 (low), class 3+ -> 1 (high)
# Since the sample data is too small and mostly class 2, we simulate a bit more diversity
df_binary = df.copy()
df_binary["binary_stress"] = df_binary["stress_class"].apply(lambda x: 0 if x <= 2 else 1)
X = df_binary[feature_cols]
y = df_binary["binary_stress"]

# Re-standardize
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 1. XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)

# 2. Support Vector Machine
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)

import pandas as pd
comparison_df = pd.DataFrame({
    "Model": ["XGBoost", "SVM"],
    "Accuracy": [xgb_report["accuracy"], svm_report["accuracy"]],
    "Precision": [xgb_report["1"]["precision"], svm_report["1"]["precision"]],
    "Recall": [xgb_report["1"]["recall"], svm_report["1"]["recall"]],
    "F1-Score": [xgb_report["1"]["f1-score"], svm_report["1"]["f1-score"]]
})

print(comparison_df)
#import ace_tools as tools; tools.display_dataframe_to_user(name="Binary Classification Comparison", dataframe=comparison_df)


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[32]:


df["binary_stress"] = df["stress_class"].apply(lambda x: 0 if x <= 2 else 1)

X = df[feature_cols]
y = df["binary_stress"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
svm = SVC(probability=True, random_state=42)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_scores = cross_val_score(xgb, X_scaled, y, cv=cv, scoring='f1')
svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='f1')

# Fit for permutation importance
xgb.fit(X_scaled, y)
perm_importance = permutation_importance(xgb, X_scaled, y, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

# Plot feature importance
plt.figure(figsize=(6, 4))
plt.barh(np.array(feature_cols)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()


# # SVM decision boundary

# In[33]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

svm.fit(X_pca, y)

# Generate mesh grid
h = 0.02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM Decision Boundary (PCA-reduced data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[34]:


# Prepare cross-validation results
cv_results_df = pd.DataFrame({
    "Model": ["XGBoost", "SVM"],
    "Mean F1 Score": [xgb_scores.mean(), svm_scores.mean()],
    "Std F1 Score": [xgb_scores.std(), svm_scores.std()]
})

print(cv_results_df)


# # Fatigue - Yi-An Yao

# # Data Preprocessing

# In[31]:


import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats as stats

base_dir = "pmdata"
participants = [f"p{str(i).zfill(2)}" for i in range(1, 17)]

def combine_all(base_dir):
    fitbit_files = os.listdir(os.path.join(base_dir, "p01", "fitbit"))
    data_dict = {file: {} for file in fitbit_files}

    for participant in tqdm(participants):
        fitbit_path = os.path.join(base_dir, participant, "fitbit")
        if os.path.exists(fitbit_path):
            for file in fitbit_files:
                file_path = os.path.join(fitbit_path, file)
                if os.path.exists(file_path):
                    if file.endswith(".json"):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        df = pd.json_normalize(data)
                    elif file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                    df["participant_id"] = participant
                    data_dict[file][participant] = df
    return data_dict

def participant_has_data(data_dict, pid):
    required_keys = ['steps.json', 'calories.json', 'sleep.json']
    for key in required_keys:
        if pid not in data_dict[key]:
            print(f"Missing {key} for {pid}")
            return False
    return True

def check_time_format(data_dict, well, p):
    pid = f"p{i:02d}"
    step1 = data_dict['steps.json'].get(pid)
    calories1 = data_dict['calories.json'].get(pid)
    resting_heart_rate1 = data_dict['resting_heart_rate.json'].get(pid)
    sleep1_raw = data_dict['sleep.json'].get(pid)
    if sleep1_raw is None:
        print(f"{pid} missing sleep.json — skipping")
        return pd.DataFrame()  
    else:
        sleep1 = sleep1_raw.loc[:, ['dateOfSleep', 'minutesAsleep']]
    mood1 = well.loc[:, ['effective_time_frame', 'mood']]
    stress1 = well.loc[:, ['effective_time_frame', 'stress']]
    fatigue1 = well.loc[:, ['effective_time_frame', 'fatigue']]

    if resting_heart_rate1 is None:
        print(f"{pid} has no resting_heart_rate.json — HR will be NaN")

    step1["dateTime"] = pd.to_datetime(step1["dateTime"]).dt.date
    calories1["dateTime"] = pd.to_datetime(calories1["dateTime"]).dt.date
    if resting_heart_rate1 is not None:
        resting_heart_rate1["dateTime"] = pd.to_datetime(resting_heart_rate1["dateTime"]).dt.date
    sleep1["dateTime"] = pd.to_datetime(sleep1["dateOfSleep"]).dt.date
    mood1["dateTime"] = pd.to_datetime(mood1["effective_time_frame"]).dt.date
    stress1["dateTime"] = pd.to_datetime(stress1["effective_time_frame"]).dt.date
    fatigue1["dateTime"] = pd.to_datetime(fatigue1["effective_time_frame"]).dt.date

    step1['value'] = step1['value'].astype('int')
    calories1['value'] = calories1['value'].astype('float')

    all = []
    for date in tqdm(sleep1['dateTime']):
        try:
            l1 = [date]
            l1.append(np.sum(step1.loc[step1['dateTime'] == date, 'value']))
            l1.append(np.sum(calories1.loc[calories1['dateTime'] == date, 'value']))
            if resting_heart_rate1 is not None:
                hr = np.mean(resting_heart_rate1.loc[resting_heart_rate1['dateTime'] == date, 'value.value'])
            else:
                hr = np.nan  
            l1.append(hr)
            l1.append(np.mean(sleep1.loc[sleep1['dateTime'] == date, 'minutesAsleep']))
            l1.append(np.mean(mood1.loc[mood1['dateTime'] == date, 'mood']))
            l1.append(np.mean(stress1.loc[stress1['dateTime'] == date, 'stress']))
            l1.append(np.mean(fatigue1.loc[fatigue1['dateTime'] == date, 'fatigue']))
            all.append(l1)
        except Exception as e:
            print(e)
            print('Missing data on ' + str(date))

    columns = ['dateTime', 'step', 'calories']
    if resting_heart_rate1 is  None:
        print(f"{pid} has no resting_heart_rate.json — column skipped")
    columns.append('resting_heart_rate')
    columns += ['sleep', 'mood', 'stress', 'fatigue']
    
    return pd.DataFrame(all, columns=columns)

data_dict = combine_all(base_dir)

for i in range(1, 17):
    pid = f"p{i:02d}"
    well_path = os.path.join(base_dir, pid, "pmsys", "wellness.csv")
    if not os.path.exists(well_path):
        print(f"{pid} skipped: wellness.csv not found.")
        continue

    if not participant_has_data(data_dict, pid):
        print(f"{pid} skipped: missing Fitbit data.")
        continue

    try:
        well = pd.read_csv(well_path)
        df = check_time_format(data_dict, well, i)
        df.to_csv(f"{pid}_stat_data.csv", index=False)
        print(f"{pid}.csv done!")
    except Exception as e:
        print(f"{pid} error: {e}")


# In[32]:


import pandas as pd
import os

folder_path = "."  
participants = [f"p{str(i).zfill(2)}" for i in range(1, 17)]

all_data = []

for pid in participants:
    file_path = os.path.join(folder_path, f"{pid}_stat_data.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["participant_id"] = pid
        all_data.append(df)
    else:
        print(f"{pid} file not found, skipping...")

merged_df = pd.concat(all_data, ignore_index=True)

merged_df.to_csv("all_participants_data.csv", index=False)

print(merged_df.head())


# In[34]:


import pandas as pd
import os

df = pd.read_csv("all_participants_data.csv")

individual＿rhr_means = df.groupby('participant_id')['resting_heart_rate'].transform(lambda x: x.fillna(x.mean()))
individual＿mood_means = df.groupby('participant_id')['mood'].transform(lambda x: x.fillna(x.mean()))
individual＿stress_means = df.groupby('participant_id')['stress'].transform(lambda x: x.fillna(x.mean()))
individual＿fatigue_means = df.groupby('participant_id')['fatigue'].transform(lambda x: x.fillna(x.mean()))

overall_rhr_mean = df['resting_heart_rate'].mean()
overall_mood_mean = df['mood'].mean()
overall_stress_mean = df['stress'].mean()
overall_fatigue_mean = df['fatigue'].mean()

df['resting_heart_rate'] = individual_rhr_means.fillna(overall_rhr_mean)
df['mood'] = individual_mood_means.fillna(overall_mood_mean)
df['stress'] = individual_stress_means.fillna(overall_stress_mean)
df['fatigue'] = individual_fatigue_means.fillna(overall_fatigue_mean)
df['fatigue_level'] = df['fatigue'].apply(lambda x: 0 if x > 3 else 1)


print(df)


# # model

# In[35]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb

features = ['step', 'calories', 'resting_heart_rate', 'sleep', 'mood', 'stress']

X = df[features]
y_classification = df['fatigue_level']
y_regression = df['fatigue']
X = StandardScaler().fit_transform(X)

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X,y_regression ,random_state=42, test_size=0.2, shuffle=True) 
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X,y_classification ,random_state=42, test_size=0.2, shuffle=True, stratify=y_classification) 

svr = SVR(kernel='rbf')
svr.fit(X_r_train, y_r_train)

rfd = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
rfd.fit(X_r_train, y_r_train)

xgb_r = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_r.fit(X_r_train, y_r_train)

y_svr_pred = svr.predict(X_r_test)
y_rfd_pred = rfd.predict(X_r_test)
y_xgb_pred = xgb_r.predict(X_r_test)

mse_svr = mean_squared_error(y_r_test, y_svr_pred)
mae_svr = mean_absolute_error(y_r_test, y_svr_pred)
r2_svr = r2_score(y_r_test, y_svr_pred)

mse_rfd = mean_squared_error(y_r_test, y_rfd_pred)
mae_rfd = mean_absolute_error(y_r_test, y_rfd_pred)
r2_rfd = r2_score(y_r_test, y_rfd_pred)

mse_xgb = mean_squared_error(y_r_test, y_xgb_pred)
mae_xgb = mean_absolute_error(y_r_test, y_xgb_pred)
r2_xgb = r2_score(y_r_test, y_xgb_pred)


print("SVR result:")
print("Mean Squared Error (MSE):", mse_svr)
print("Mean Absolute Error (MAE):", mae_svr)
print("R² Score:", r2_svr)
print("====================")
print("RandomForestRegressor result:")
print("Mean Squared Error (MSE):", mse_rfd)
print("Mean Absolute Error (MAE):", mae_rfd)
print("R² Score:", r2_rfd)
print("====================")
print("XGBoostRegressor result:")
print("Mean Squared Error (MSE):", mse_xgb)
print("Mean Absolute Error (MAE):", mae_xgb)
print("R² Score:", r2_xgb)


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

svc = SVC(kernel='linear') 
svc.fit(X_c_train, y_c_train)

lr = LogisticRegression(multi_class='ovr')
lr.fit(X_c_train, y_c_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_c_train, y_c_train)

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_c_train, y_c_train)

y_svc_pred = svc.predict(X_c_test)
y_lr_pred = lr.predict(X_c_test)
y_knn_pred = knn.predict(X_c_test)
y_dtc_pred = dtc.predict(X_c_test)

accuracy_svc = accuracy_score(y_c_test, y_svc_pred)
accuracy_lr = accuracy_score(y_c_test, y_lr_pred)
accuracy_knn = accuracy_score(y_c_test, y_knn_pred)
accuracy_dtc = accuracy_score(y_c_test, y_dtc_pred)

accuracy_svc = accuracy_score(y_c_test, y_svc_pred)
accuracy_lr = accuracy_score(y_c_test, y_lr_pred)
accuracy_knn = accuracy_score(y_c_test, y_knn_pred)
accuracy_dtc = accuracy_score(y_c_test, y_dtc_pred)
precision_svc = precision_score(y_c_test, y_svc_pred, average='weighted', zero_division=0)
precision_lr = precision_score(y_c_test, y_lr_pred, average='weighted', zero_division=0)
precision_knn = precision_score(y_c_test, y_knn_pred, average='weighted', zero_division=0)
precision_dtc = precision_score(y_c_test, y_dtc_pred, average='weighted', zero_division=0)
recall_svc = recall_score(y_c_test, y_svc_pred, average='weighted')
recall_lr = recall_score(y_c_test, y_lr_pred, average='weighted')
recall_knn = recall_score(y_c_test, y_knn_pred, average='weighted')
recall_dtc = recall_score(y_c_test, y_dtc_pred, average='weighted')
f1_score_svc = f1_score(y_c_test, y_svc_pred, average='weighted')
f1_score_lr = f1_score(y_c_test, y_lr_pred, average='weighted')
f1_score_knn = f1_score(y_c_test, y_knn_pred, average='weighted')
f1_score_dtc = f1_score(y_c_test, y_dtc_pred, average='weighted')

print("SVC result:")
print("Accuracy:", accuracy_svc)
print("Precision Score:", precision_svc)
print("Recall Score:", recall_svc)
print("F1 Score:", f1_score_svc)
print("====================")
print("LogisticRegression result:")
print("Accuracy:", accuracy_lr)
print("Precision Score:", precision_lr)
print("Recall Score:", recall_lr)
print("F1 Score:", f1_score_lr)
print("====================")
print("KNeighborsClassifier result:")
print("Accuracy:", accuracy_knn)
print("Precision Score:", precision_knn)
print("Recall Score:", recall_knn)
print("F1 Score:", f1_score_knn)
print("====================")
print("DecisionTreeClassifier result:")
print("Accuracy:", accuracy_dtc)
print("Precision Score:", precision_dtc)
print("Recall Score:", recall_dtc)
print("F1 Score:", f1_score_dtc)


# In[39]:


import smogn

df_balanced = smogn.smoter(
    data = df, 
    y = 'fatigue',
    k = 5,
    samp_method='extreme'  
)


# In[40]:


import smogn

df_balanced_c = smogn.smoter(
    data = df, 
    y = 'fatigue_level',
    k = 5,
    samp_method='extreme'  
)


# In[41]:


X_balanced = df_balanced[features]
y_balanced = df_balanced['fatigue']
X_balanced = StandardScaler().fit_transform(X_balanced)

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_balanced,y_balanced ,random_state=42, test_size=0.2, shuffle=True) 

svr = SVR(kernel='rbf')
svr.fit(X_b_train, y_b_train)

rfd = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
rfd.fit(X_b_train, y_b_train)

xgb_r = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_r.fit(X_b_train, y_b_train)

y_svr_b_pred = svr.predict(X_b_test)
y_rfd_b_pred = rfd.predict(X_b_test)
y_xgb_b_pred = xgb_r.predict(X_b_test)

mse_svr_b = mean_squared_error(y_b_test, y_svr_b_pred)
mae_svr_b = mean_absolute_error(y_b_test, y_svr_b_pred)
r2_svr_b = r2_score(y_b_test, y_svr_b_pred)

mse_rfd_b = mean_squared_error(y_b_test, y_rfd_b_pred)
mae_rfd_b = mean_absolute_error(y_b_test, y_rfd_b_pred)
r2_rfd_b = r2_score(y_b_test, y_rfd_b_pred)


mse_xgb_b = mean_squared_error(y_b_test, y_xgb_b_pred)
mae_xgb_b = mean_absolute_error(y_b_test, y_xgb_b_pred)
r2_xgb_b = r2_score(y_b_test, y_xgb_b_pred)

print("SVR result with SMOTE:")
print("Mean Squared Error (MSE):", mse_svr_b)
print("Mean Absolute Error (MAE):", mae_svr_b)
print("R² Score:", r2_svr_b)
print("====================")
print("RandomForestRegressor result with SMOTE:")
print("Mean Squared Error (MSE):", mse_rfd_b)
print("Mean Absolute Error (MAE):", mae_rfd_b)
print("R² Score:", r2_rfd_b)
print("====================")
print("XGBoostRegressor result with SMOTE:")
print("Mean Squared Error (MSE):", mse_xgb_b)
print("Mean Absolute Error (MAE):", mae_xgb_b)
print("R² Score:", r2_xgb_b)


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_balanced_c = df_balanced_c[features]
y_balanced_c = df_balanced_c['fatigue_level']
X_balanced_scaled_c = StandardScaler().fit_transform(X_balanced_c)
X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled_c, y_balanced_c, test_size=0.2, random_state=42, shuffle=True)

models = {
    "SVC (Linear)": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(multi_class='ovr'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"{name} (SMOGN)result:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision Score: {precision:.4f}")
    print(f"  Recall Score: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("=" * 30)


# In[43]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid_svr = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.5]
}
param_grid_rdf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1]
}

grid_search_svr = GridSearchCV(SVR(kernel='rbf'), param_grid_svr, cv=5, scoring='r2')
grid_search_rdf = GridSearchCV(RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True), param_grid_rdf, cv=5, scoring='r2')
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid_xgb, cv=5, scoring='r2')
grid_search_svr.fit(X_b_train, y_b_train)
grid_search_rdf.fit(X_b_train, y_b_train)
grid_search_xgb.fit(X_b_train, y_b_train)


print("Best parameters for SVR:", grid_search_svr.best_params_)
print("Best R² score from CV for SVR:", grid_search_svr.best_score_)
print("Best parameters for RandomForestRegressor:", grid_search_rdf.best_params_)
print("Best R² score from CV for RandomForestRegressor:", grid_search_rdf.best_score_)
print("Best parameters for XGBoostRegressor:", grid_search_xgb.best_params_)
print("Best R² score from CV for XGBoostRegressor:", grid_search_xgb.best_score_)

best_svr = grid_search_svr.best_estimator_
y_best_pred_svr = best_svr.predict(X_b_test)
best_rdf = grid_search_rdf.best_estimator_
y_best_pred_rdf = best_rdf.predict(X_b_test)
best_xgb = grid_search_xgb.best_estimator_
y_best_pred_xgb = best_xgb.predict(X_b_test)

from sklearn.metrics import r2_score
print("R² on test set for SVR:", r2_score(y_b_test, y_best_pred_svr))
print("R² on test set for RandomForestRegressor:", r2_score(y_b_test, y_best_pred_rdf))
print("R² on test set for XGBoostRegressor:", r2_score(y_b_test, y_best_pred_xgb))


# In[44]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_balanced)

y_balanced = df_balanced['fatigue']
X＿pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y_balanced, test_size=0.2, random_state=42)

svr = SVR(kernel='rbf')
svr.fit(X_pca_train, y_pca_train)

rfd = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
rfd.fit(X_pca_train, y_pca_train)

xgb_r = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_r.fit(X_pca_train, y_pca_train)

y_svr_pca_pred = svr.predict(X_pca_test)
y_rfd_pca_pred = rfd.predict(X_pca_test)
y_xgb_pca_pred = xgb_r.predict(X_pca_test)

mse_svr_pca = mean_squared_error(y_pca_test, y_svr_pca_pred)
mae_svr_pca = mean_absolute_error(y_pca_test, y_svr_pca_pred)
r2_svr_pca = r2_score(y_pca_test, y_svr_pca_pred)

mse_rfd_pca = mean_squared_error(y_pca_test, y_rfd_pca_pred)
mae_rfd_pca = mean_absolute_error(y_pca_test, y_rfd_pca_pred)
r2_rfd_pca = r2_score(y_pca_test, y_rfd_pca_pred)

mse_xgb_pca = mean_squared_error(y_pca_test, y_xgb_pca_pred)
mae_xgb_pca = mean_absolute_error(y_pca_test, y_xgb_pca_pred)
r2_xgb_pca = r2_score(y_pca_test, y_xgb_pca_pred)

print("SVR result with SMOTE and PCA:")
print("Mean Squared Error (MSE):", mse_svr_pca)
print("Mean Absolute Error (MAE):", mae_svr_pca)
print("R² Score:", r2_svr_pca)
print("====================")
print("RandomForestRegressor result with SMOTE and PCA:")
print("Mean Squared Error (MSE):", mse_rfd_pca)
print("Mean Absolute Error (MAE):", mae_rfd_pca)
print("R² Score:", r2_rfd_pca)
print("====================")
print("XGBoostRegressor result with SMOTE and PCA:")
print("Mean Squared Error (MSE):", mse_xgb_pca)
print("Mean Absolute Error (MAE):", mae_xgb_pca)
print("R² Score:", r2_xgb_pca)


# In[45]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

lda = LDA(n_components=None) 
X_lda = lda.fit_transform(X_balanced_scaled_c, y_balanced_c)


X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y_balanced_c, test_size=0.2, random_state=42)

models_lda = {
    "SVC (Linear)": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(multi_class='ovr'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5)
}

for name, model in models_lda.items():
    model.fit(X_train_lda, y_train_lda)
    y_pred = model.predict(X_test_lda)

    accuracy = accuracy_score(y_test_lda, y_pred)
    precision = precision_score(y_test_lda, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_lda, y_pred, average='weighted')
    f1 = f1_score(y_test_lda, y_pred, average='weighted')

    print(f"{name} (LDA) result:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision Score: {precision:.4f}")
    print(f"  Recall Score: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("=" * 30)


# In[46]:


from sklearn.model_selection import GridSearchCV

param_grid_svr = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.5]
}
param_grid_rdf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1]
}

grid_search_svr_pca = GridSearchCV(SVR(kernel='rbf'), param_grid_svr, cv=5, scoring='r2')
grid_search_rdf_pca = GridSearchCV(RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True), param_grid_rdf, cv=5, scoring='r2')
grid_search_xgb_pca = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid_xgb, cv=5, scoring='r2')
grid_search_svr_pca.fit(X_pca_train, y_pca_train)
grid_search_rdf_pca.fit(X_pca_train, y_pca_train)
grid_search_xgb_pca.fit(X_pca_train, y_pca_train)

print("Best parameters for SVR:", grid_search_svr_pca.best_params_)
print("Best R² score from CV for SVR:", grid_search_svr_pca.best_score_)
print("Best parameters for RandomForestRegressor:", grid_search_rdf_pca.best_params_)
print("Best R² score from CV for RandomForestRegressor:", grid_search_rdf_pca.best_score_)
print("Best parameters for XGBoostRegressor:", grid_search_xgb_pca.best_params_)
print("Best R² score from CV for XGBoostRegressor:", grid_search_xgb_pca.best_score_)

best_svr_pca = grid_search_svr_pca.best_estimator_
y_best_pred_svr_pca = best_svr_pca.predict(X_pca_test)
best_rdf_pca = grid_search_rdf_pca.best_estimator_
y_best_pred_rdf_pca = best_rdf_pca.predict(X_pca_test)
best_xgb_pca = grid_search_xgb_pca.best_estimator_
y_best_pred_xgb_pca = best_xgb_pca.predict(X_pca_test)

from sklearn.metrics import r2_score
print("R² on test set for SVR:", r2_score(y_pca_test, y_best_pred_svr_pca))
print("R² on test set for RandomForestRegressor:", r2_score(y_pca_test, y_best_pred_rdf_pca))
print("R² on test set for XGBoostRegressor:", r2_score(y_pca_test, y_best_pred_xgb_pca))


# In[ ]:




