# 这个代码干的事情其实就是用梯度提升回归树 (GradientBoostingRegressor, GBDT) 对地理网格数据建模，并做特征重要性分析 + 偏依赖图 (PDP) 提取。
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from scipy.stats import randint, uniform, loguniform

# 文件夹路径
grid_folder = r'D:\seoul\grids\lst_map\final_clean\480_based'
def run_gbdt(grid_folder, years=[2016, 2023], n=0):
    for year in years:
        target_vars = [f'nor_{year}', f'ext_{year}', f'hr_{year}']
        explanatory_vars = ['BCR', 'BHV',  'SVF', 'NDVI', 'EV', 'WR', 'Dist_W', 'Dist_P', 'Dist_M','X','Y'] # 顺序很讲究
        # 保存结果
        all_results = []
        pdp_records = []
        r2_comparison = []

        param_dist = {
            'n_estimators': [4168], #4168
            'learning_rate': loguniform(0.002, 0.355), #(0.002, 0.355)
            'subsample': uniform(0.545, 0.413), # [0.545,0.958]
            'max_depth' : randint(5, 14), # [5, 13]
            'min_samples_split':[2], #2
            'max_features': uniform(0.335, 0.581), #[0.335,0.916]
            }

        # === 主循环 ===
        for filename in os.listdir(grid_folder):
            if filename.endswith(f'city{year}_lst_ratio_grid_480m_bcr_bhv_ndvi_svf_ev_distbp_distmt_distwb_wr_xy.shp'):
                input_path = os.path.join(grid_folder, filename)
                match = re.search(r'(\d{3,5})m', filename)
                grid_size = match.group(1)

                gdf = gpd.read_file(input_path)
                gdf_clean = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=target_vars + explanatory_vars)

                for target in target_vars:
                    X = gdf_clean[explanatory_vars]
                    y = gdf_clean[target]

                    #####################################################################
                    # 用20个 random seeds
                    np.random.seed(0)  # 固定种子以便复现
                    random_seeds = np.random.choice(10000, size=20, replace=False)

                    # print(random_seeds)
                    n = n
                    #####################################################################
                    for r in [random_seeds[n]]:

                        # 数据划分
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r) # 切20次

                        gbdt = GradientBoostingRegressor(random_state=0)
                        
                        cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)

                        search = RandomizedSearchCV(
                            estimator=gbdt,
                            param_distributions=param_dist,
                            n_iter= 200,
                            scoring='r2',
                            cv=cv, # cross validation
                            verbose=2,
                            n_jobs=-1,
                            random_state=0
                        )

                        search.fit(X_train, y_train)

                        results_df = pd.DataFrame(search.cv_results_)
                        results_df.to_csv(folder, rf"{n}_{r}_GBDT_{target}_cv_results.csv", index=False)

                        # 使用测试集评估
                        best_model = search.best_estimator_
                        y_train_pred = best_model.predict(X_train)
                        y_test_pred = best_model.predict(X_test)

                        r2_train = best_model.score(X_train, y_train)
                        r2_test = r2_score(y_test, y_test_pred)

                        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
                        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

                        print(f" {filename} | {target} 最佳参数: {search.best_params_} | R²_train={r2_train:.3f} | R²_test={r2_test:.3f}")

                        for var, importance in zip(explanatory_vars, best_model.feature_importances_):
                            all_results.append({
                                'GridSize': grid_size,
                                'Target': target,
                                'Feature': var,
                                'Random seed':r,
                                'FeatureImportance_TrainModel': round(importance, 4),
                                'Train_R2': round(r2_train, 4),
                                'Train_RMSE': round(rmse_train, 4),
                                'Test_R2': round(r2_test, 4),
                                'Test_RMSE': round(rmse_test, 4),
                                **search.best_params_
                            })
                            r2_comparison.append({
                                'GridSize': grid_size,
                                'Target': target,
                                'Random seed':r,
                                'Train_R2': round(r2_train, 4),
                                'Test_R2': round(r2_test, 4),
                                'Train_RMSE': round(rmse_train, 4),
                                'Test_RMSE': round(rmse_test, 4)
                            })
                

        # 保存模型训练后的结果
        df_all = pd.DataFrame(all_results)
        folder = os.path.join(grid_folder, r'Machine Learning')
        os.makedirs(folder,exist_ok=True)
        df_all.to_excel(os.path.join(folder, f'{year} GBDT_Random_Search_Results.xlsx'), index=False)
        df_r2 = pd.DataFrame(r2_comparison)
        df_r2 = df_r2.sort_values(['Target', 'GridSize'])
        df_r2.to_excel(os.path.join(folder, f'{year} R2_Comparison_Train_vs_Test.xlsx'), index=False)
        # print("✅ R² train vs test comparison saved.")