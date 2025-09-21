# main_GBDT.py
from joblib import Parallel, delayed
from GBDT import run_gbdt

if __name__ == "__main__":
    grid_folder = r"D:\seoul\grids\lst_map\final_clean\480_based"
    seeds = range(20)  # 对应 n=0,1,...,19

    # 并行执行，每个 seed 跑一次
    Parallel(n_jobs=-4)(   # n_jobs=-1 表示用满所有CPU
        delayed(run_gbdt)(grid_folder, years=[2016, 2023], n=i)
        for i in seeds
    )
    
