import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('df2.csv',encoding='gbk')
#print(data.columns)

#change the name
column_mapping = {
    '是否新辅助治疗': 'new_treat',
    '是否化疗': 'new_chem',
    '是否放疗': 'new_rad',
    '免疫治疗': 'new_imu',
    '次数\n实际完成次数': 'new_cnt',
    '是否术后辅助治疗': 'post_treat',
    '术后化疗方案': 'post_chem',
    '术后放疗方案': 'post_rad',
    '靶向药物.1': 'post_target',
    '免疫治疗.1': 'post_imu',
    '术后治疗\n实际完成次数': 'post_cnt',
    '是否进展': 'progress',
    '是否死亡': 'death',
    '复发后是否治疗': 're_treat',
    '化疗方案': 're_chem',
    '放疗方案': 're_rad',
    '手术方案': 're_ope',
    '靶向药物\n（可多选）': 're_target',
    '免疫治疗.2': 're_imu',
    '复发后治疗\n实际完成次数': 're_cnt',
    'DFS（days）': 'dfs'
}

data.rename(columns=column_mapping, inplace=True)

print(data.columns)
target = data['dfs']
data.drop(columns=['dfs'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


rf_model =  RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
feature_importance = rf_model.feature_importances_
features = X_train.columns

import joblib
joblib.dump(rf_model, 'model/model.pkl')

print(rf_model.get_params())

# 按照特征重要度从大到小排序
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]
sorted_features = features[sorted_indices]

#使用SHAP进行可视化
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei'] 
shap.summary_plot(shap_values, X_train, feature_names=sorted_features, plot_type='bar')
plt.show()
