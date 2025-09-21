import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 设置文件夹、变量
grid_folder = r'D:\seoul\grids\lst_map\final_clean\480_based'
output_fig_dir = r'D:\seoul\grids\lst_map\final_clean\480_based\Machine Learning\figures\nonlinear_predictions'
os.makedirs(output_fig_dir, exist_ok=True)

explanatory_vars = ['BCR', 'BHV', 'SVF', 'NDVI', 'EV', 'WR', 'Dist_W', 'Dist_P', 'Dist_M']
explanatory_vars_clean = ['BCR', 'BHV', 'SVF', 'NDVI', 'EV', 'WR']

years = [2016, 2023]
target_vars_template = ['nor_{}', 'ext_{}', 'hr_{}']

for year in years:
    # 清理后的网格数据
    file  = fr'{grid_folder}\city{year}_lst_ratio_grid_480m_bcr_bhv_ndvi_svf_ev_distbp_distmt_distwb_wr_xy.shp'
    gdf_clean = gpd.read_file(file).replace([np.inf, -np.inf], np.nan)
    
    # 读取最佳参数
    best_params_file_gbdt = rf'{grid_folder}\Machine Learning\{year} GBDT_Random_Search_Results.xlsx'
    best_df_gbdt = pd.read_excel(best_params_file_gbdt)
    best_params_file_sdem = rf'{grid_folder}\statistics\SDEM_all_params.xlsx'
    best_df_sdem = pd.read_excel(best_params_file_sdem)

    # 输出文件夹
    output_dir = os.path.join(output_fig_dir, f'{year}')
    os.makedirs(output_dir, exist_ok=True)
    
    for target in best_df_gbdt['Target'].unique():
        
        # 取第一行参数
        row_gbdt = best_df_gbdt[best_df_gbdt['Target']==target].iloc[0]
        params_gbdt = {
            'learning_rate': row_gbdt['learning_rate'],
            'max_depth': int(row_gbdt['max_depth']),
            'n_estimators': int(row_gbdt['n_estimators']),
            'subsample': row_gbdt['subsample'],
            'min_samples_split': int(row_gbdt['min_samples_split']),
            'max_features': float(row_gbdt['max_features']),
            'random_state': 0
        }
        row_sdem = best_df_sdem[best_df_sdem['Target']==target].iloc[0]
        params_sdem = {
            'CONSTANT': row_sdem['CONSTANT'],
            'BCR': row_sdem['BCR'],
            'BHV': row_sdem['BHV'],
            'SVF': row_sdem['SVF'],
            'NDVI': row_sdem['NDVI'],
            'EV': row_sdem['EV'],
            'WR': row_sdem['WR'],
            'Dist_W': row_sdem['Dist_W'],
            'Dist_P': row_sdem['Dist_P'],
            'Dist_M': row_sdem['Dist_M'],
            'W_BCR': row_sdem['W_BCR'],
            'W_BHV': row_sdem['W_BHV'],
            'W_SVF': row_sdem['W_SVF'],
            'W_NDVI': row_sdem['W_NDVI'],
            'W_EV': row_sdem['W_EV'],
            'W_WR': row_sdem['W_WR'],
            'lambda': row_sdem['lambda']
        }

        # 训练 GBDT
        X = gdf_clean[explanatory_vars]
        y = gdf_clean[target]
        model = GradientBoostingRegressor(**params_gbdt)
        model.fit(X, y)
        
        # ------------------- 同一张图绘制 -------------------
        for feature in explanatory_vars:

            fig_path = os.path.join(output_dir, f'{target}_{feature}_comparison.png')
            # if os.path.exists(fig_path):
            #     continue

            n_points = 100
            X_grid = np.linspace(X[feature].min(), X[feature].max(), n_points)

            ######################## GBDT ####################################
            # 所有的variable.其他几个variables不变。2000个点。0.05->0.8 BCR. 
            X_base = pd.DataFrame(np.tile(X.mean().values, (n_points,1)), columns=explanatory_vars)
            X_base[feature] = X_grid
            y_gbdt_pred = model.predict(X_base)
            # print(X_base)
            # print(X_base[feature])
            # print(y_gbdt_pred)
            # print(y_gbdt_pred.shape)

            ######################## SDEM ####################################
            X_mean = gdf_clean[explanatory_vars].mean().to_dict()
            w = DistanceBand.from_dataframe(gdf_clean, threshold=1000, binary=False)
            w.transform = 'r'

            WX = lag_spatial(w, gdf_clean[explanatory_vars_clean].values)
            WX_mean = pd.Series(WX.mean(axis=0), index=[f"W_{v}" for v in explanatory_vars_clean])

            # 预测结果
            y_sdem_pred = []
            for val in X_grid:   # 每次只改一个 feature
                X_temp = X_mean.copy()
                X_temp[feature] = val
                all_vars = {'CONSTANT': 1.0, **X_temp, **WX_mean.to_dict()}
                y_pred = 0
                for var, coef in params_sdem.items():
                    if var != 'lambda': 
                        y_pred += coef * all_vars.get(var, 0)
                y_sdem_pred.append(y_pred)

            y_sdem_pred = np.array(y_sdem_pred)

            # ------------------- 绘图 -------------------
            fig = plt.subplots(figsize=(8,5))
            # GBDT 预测曲线（蓝色）
            plt.plot(X_grid, y_gbdt_pred, color='#333333', linewidth=2, label='GBDT prediction')

            # SDEM 单变量趋势（红色虚线）
            plt.plot(X_grid, y_sdem_pred, color="#4000ff", linestyle='--', linewidth=2, label='SDEM trend')
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.title(f'{feature} effect on {target}')
            plt.grid(True)
            plt.legend()
            # 固定 y 轴
            if 'hr' in target:
                plt.ylim(-20, 0)
            elif 'ext' in target:
                plt.ylim(25, 55)
            # else:
            #     plt.ylim(20, 40)
            else:
                plt.ylim(20, 50)
            plt.tight_layout()
            

            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"✅ Saved {fig_path}")