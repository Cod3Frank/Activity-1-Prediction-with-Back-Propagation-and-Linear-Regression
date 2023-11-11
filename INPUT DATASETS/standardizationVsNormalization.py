import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("dataset.csv").drop('target', axis = 1)
min_max_scaler = MinMaxScaler()
std_scaler = StandardScaler()
# MinMaxScaler() class fit_transform function first fits on data and then transform the given data
# and returns array, therefore we have to construct dataframe for normalized data
data_normalized = min_max_scaler.fit_transform(data)
df_data_normalized = pd.DataFrame(data_normalized, columns= list(data.columns))
# StandardScaler() class fit_transform function first fits on data and then transform the given data
# and returns array, therefore we have to construct dataframe for standardized data
data_standardized = std_scaler.fit_transform(data)
df_data_standardized = pd.DataFrame(data_standardized, columns= list(data.columns))
plt.figure(figsize = (20,20))
for idx, col in enumerate(list(data.columns)):
    plt.subplot(5,3,idx + 1)
    sns.kdeplot(data[col], color = 'g',label = f"Input data(Unscaled):{col}")
    sns.kdeplot(df_data_normalized[col],color = 'b', label = f"Input data(Normalized):{col}" )
    sns.kdeplot(df_data_standardized[col],color = 'r', label = f"Input data(Standardized):{col}" )
    plt.legend()
plt.show()
