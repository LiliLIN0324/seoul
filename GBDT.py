from libpysal.weights import DistanceBand, lag_spatial
from spreg import ML_Lag
import pandas as pd
import numpy as np
import geopandas as gpd
import os

def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''

grid_folder = r'D:\seoul\grids\lst_map\final_clean\480_based'
output_fig_dir = r'D:\seoul\grids\lst_map\final_clean\480_based\statistics\fig\figures'
output_file = r'D:\seoul\grids\lst_map\final_clean\480_based\statistics\SDEM_effects.xlsx'

for year in [2016, 2023]:
    target_vars = [f'nor_{year}', f'ext_{year}', f'hr_{year}']
    explanatory_vars = ['BCR', 'BHV',  'SVF', 'NDVI', 'EV', 'WR', 'Dist_P', 'Dist_M', 'Dist_W']
    explanatory_vars_clean = ['BCR', 'BHV',  'SVF', 'NDVI', 'EV', 'WR']
    results_by_target = {t: [] for t in target_vars}
    # 使用 ExcelWriter，mode='a' 可以追加 sheet（如果文件存在）
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a' if os.path.exists(output_file) else 'w') as writer:

        for filename in os.listdir(grid_folder):
            if filename.endswith('480m_bcr_bhv_ndvi_svf_ev_distbp_distmt_distwb_wr_xy.shp') and filename.startswith(f'city{year}'):
                path = os.path.join(grid_folder, filename)
                gdf = gpd.read_file(path).replace([np.inf, -np.inf], np.nan)

                for target in target_vars:
                    if target not in gdf.columns:
                        continue

                    data = gdf[explanatory_vars + [target]].dropna()
                    if data.empty:
                        continue

                    data_gdf = gdf.loc[data.index]
                    w = DistanceBand.from_dataframe(data_gdf, threshold=1000, binary=False)
                    w.transform = 'r'

                    # y
                    y = data[target].values.reshape(-1, 1)

                    # X: 原变量
                    X_main = data[explanatory_vars].values
                    # print(explanatory_vars)

                    # WX_clean: 只 lag clean 变量
                    WX_clean = lag_spatial(w, data[explanatory_vars_clean].values)
                    # print(explanatory_vars_clean)

                    # 合并常数 + X_main + WX_clean
                    X_all = np.hstack([np.ones((len(data), 1)), X_main, WX_clean])

                    name_x = ['const'] + explanatory_vars + [f"W_{v}" for v in explanatory_vars_clean]

                    # 模型
                    model = ML_Error(
                        y, X_all,
                        w=w,
                        name_y=target,
                        name_x=name_x,
                        spat_diag=True
                    )

                    betas = model.betas.flatten()

                    # ==================================================
                    # 直接效应：原变量系数
                    direct = betas[1:1+len(explanatory_vars)]
                    # 间接效应：WX 系数
                    indirect = betas[1+len(explanatory_vars):]

                    total = []
                    for var in explanatory_vars:
                        if var in explanatory_vars_clean:
                            total.append(direct[explanatory_vars.index(var)] + 
                                        indirect[explanatory_vars_clean.index(var)])
                        else:
                            total.append(direct[explanatory_vars.index(var)])  # 没有 WX，对应间接效应为 0

                    # 构建表格
                    effects_df = pd.DataFrame({
                        "Variable": explanatory_vars,
                        "Direct": direct,
                        "Indirect": [indirect[explanatory_vars_clean.index(v)] if v in explanatory_vars_clean else 0 for v in explanatory_vars],
                        "Total": total
                    })

                    print(f"=== {target} ===")
                    print(effects_df)

                    # 保存到 Excel，不同 target 用 sheet 名
                    sheet_name = target
                    effects_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved sheet: {sheet_name}")
                    # ==================================================

                    # ====== 绘制曲线 ======
                    for var in explanatory_vars:
                        var_idx = explanatory_vars.index(var)
                        x_vals = np.linspace(data[var].min(), data[var].max(), 50)
                        y_direct = effects_df.loc[var_idx, "Direct"] * x_vals
                        y_indirect = effects_df.loc[var_idx, "Indirect"] * x_vals
                        y_total = effects_df.loc[var_idx, "Total"] * x_vals

                        plt.figure(figsize=(6,4))
                        plt.plot(x_vals, y_direct, label="Direct", color="blue")
                        plt.plot(x_vals, y_indirect, label="Indirect", color="orange")
                        plt.plot(x_vals, y_total, label="Total", color="green")
                        plt.xlabel(var)
                        plt.ylabel(f"Predicted change in {target}")
                        plt.title(f"{target} - Effect of {var}")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_fig_dir, f"{target}_{var}_effect_curve.png"), dpi=300)
                        plt.close()

                    # ====== 绘制曲线 ======
                    stats = model.z_stat

                    rows = []
                    for i, var in enumerate(name_x):
                        coef = f"{betas[i]:.4f}{star(stats[i][1])}"
                        tval = round(stats[i][0], 3)
                        pval = round(stats[i][1], 4)
                        rows.append([var, coef, tval, pval])

                    rho_coef = f"{betas[-1]:.4f}{star(stats[-1][1])}"
                    rho_t = round(stats[-1][0], 3)
                    rho_p = round(stats[-1][1], 4)
                    rows.append(["rho", rho_coef, rho_t, rho_p])

                    rows.append(["Log-likelihood", round(float(model.logll), 4), "", ""])
                    r2_val = getattr(model, 'pr2', getattr(model, 'r2', np.nan))
                    rows.append(["R²", round(float(r2_val), 4), "", ""])
                    rows.append(["", "", "", ""])

                    df_file = pd.DataFrame(rows, columns=["Feature", "Coefficient", "t-value", "p-value"])
                    results_by_target[target].append(df_file)

                    outdir = os.path.join(grid_folder, 'statistics')
                    os.makedirs(outdir, exist_ok=True)
                    out_xlsx = os.path.join(outdir, f'{year}_SDEM.xlsx')

        with pd.ExcelWriter(out_xlsx) as wtr:
            for target in target_vars:
                if not results_by_target[target]:
                    continue
                df_target = pd.concat(results_by_target[target], ignore_index=True)
                df_target.to_excel(wtr, sheet_name=target, index=False)

        print(f"✅ 已生成：{out_xlsx}")