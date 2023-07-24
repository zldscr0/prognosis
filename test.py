import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('df2.csv',encoding='gbk')
print(data.columns)


target = data['DFS（days）']
data.drop(columns=['DFS（days）'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


gb_model =  RandomForestRegressor(random_state=42)
gb_model.fit(X_train, y_train)
feature_importance = gb_model.feature_importances_
features = X_train.columns

# 按照特征重要度从大到小排序
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]
sorted_features = features[sorted_indices]

#使用SHAP进行可视化
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_train)

from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei'] 
shap.summary_plot(shap_values, X_train, feature_names=sorted_features, plot_type='bar')
plt.show()
