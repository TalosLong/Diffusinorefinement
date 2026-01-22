import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results_file(filepath):
    """解析结果TXT文件读取Metrics"""
    metrics = {}
    current_section = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("Refined Results:"):
            current_section = "Refined"
            continue
        elif line.startswith("Coarse Results:"):
            current_section = "Coarse"
            continue
        
        # 匹配 "Metric: Value" 格式
        # 假设格式: "Dice: 0.8854" 或类似
        if ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    if current_section == "Refined":
                        metrics[key] = val
                    elif current_section == "Coarse":
                        metrics[f"Coarse_{key}"] = val
                except ValueError:
                    pass
    return metrics

def analyze_grid_results(root_dir):
    data = []
    
    # 遍历目录
    subdirs = glob.glob(os.path.join(root_dir, "steps*_start*"))
    
    print(f"Found {len(subdirs)} experiment directories.")
    
    for subdir in subdirs:
        dirname = os.path.basename(subdir)
        
        # 解析参数
        try:
            steps_match = re.search(r'steps(\d+)', dirname)
            start_match = re.search(r'start(\d+)', dirname)
            
            if not steps_match or not start_match:
                continue
                
            steps = int(steps_match.group(1))
            start_t = int(start_match.group(1))
            
            # 找到结果文件
            result_files = glob.glob(os.path.join(subdir, "*_results.txt"))
            if not result_files:
                continue
                
            # 读取最新的一个
            metrics = parse_results_file(result_files[0])
            
            # 整理一行数据
            row = {
                'Steps': steps,
                'Start_T': start_t,
                'Dice': metrics.get('Dice', 0),
                'HD95': metrics.get('HD95', 100),
                'IoU': metrics.get('IoU', 0),
                'Coarse_Dice': metrics.get('Coarse_Dice', 0)
            }
            # 计算提升
            row['Dice_Gain'] = row['Dice'] - row['Coarse_Dice']
            
            data.append(row)
            
        except Exception as e:
            print(f"Error processing {dirname}: {e}")

    df = pd.DataFrame(data)
    
    # 打印前几行数据以便调试
    print("\nParsed Data Preview:")
    print(df.head())
    print("\nData Statistics:")
    print(df.describe())
    
    return df

def plot_heatmaps(df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 使用 pivot_table 并显式排序，防止轴乱序
    # index=Steps (y轴), columns=Start_T (x轴)
    pivot_dice = df.pivot_table(index="Steps", columns="Start_T", values="Dice", aggfunc='mean')
    pivot_hd95 = df.pivot_table(index="Steps", columns="Start_T", values="HD95", aggfunc='mean')
    pivot_gain = df.pivot_table(index="Steps", columns="Start_T", values="Dice_Gain", aggfunc='mean')
    
    # 确保行列排序
    pivot_dice = pivot_dice.sort_index(axis=0).sort_index(axis=1)
    pivot_hd95 = pivot_hd95.sort_index(axis=0).sort_index(axis=1)
    pivot_gain = pivot_gain.sort_index(axis=0).sort_index(axis=1)

    print("\nDice Matrix for Heatmap:")
    print(pivot_dice)
    
    # 1. Dice Heatmap
    sns.heatmap(pivot_dice, annot=True, fmt=".4f", cmap="magma", ax=axes[0])
    axes[0].set_title("Dice Score (Higher is Better)", fontsize=14)
    axes[0].set_xlabel("Start Timestep", fontsize=12)
    axes[0].set_ylabel("Inference Steps", fontsize=12)
    axes[0].invert_yaxis() # y轴反向：让小步数在下，大步数在上，符合直角坐标系直觉
    
    # 2. HD95 Heatmap
    sns.heatmap(pivot_hd95, annot=True, fmt=".2f", cmap="viridis_r", ax=axes[1]) # _r for reversed, low is good
    axes[1].set_title("HD95 (Lower is Better)", fontsize=14)
    axes[1].set_xlabel("Start Timestep", fontsize=12)
    axes[1].set_ylabel("Inference Steps", fontsize=12)
    axes[1].invert_yaxis()
    
    # 3. Efficiency/Gain Heatmap
    sns.heatmap(pivot_gain, annot=True, fmt=".4f", cmap="coolwarm", center=0, ax=axes[2])
    axes[2].set_title("Dice Improvement vs Coarse Baseline", fontsize=14)
    axes[2].set_xlabel("Start Timestep", fontsize=12)
    axes[2].set_ylabel("Inference Steps", fontsize=12)
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nHeatmaps saved to {output_path}")

def recommend_parameters(df):
    if df.empty:
        print("No data found to analyze.")
        return

    print("\n" + "="*50)
    print("PARAMATER SELECTION RECOMMENDATION")
    print("="*50)
    
    # 1. 最佳性能 (Max Dice)
    best_dice_row = df.loc[df['Dice'].idxmax()]
    print(f"🏆 Best Quality (Highest Dice):")
    print(f"   Steps: {int(best_dice_row['Steps'])} | Start_T: {int(best_dice_row['Start_T'])}")
    print(f"   Dice: {best_dice_row['Dice']:.4f} | HD95: {best_dice_row['HD95']:.2f}")
    
    # 2. 最佳边界 (Min HD95)
    best_hd_row = df.loc[df['HD95'].idxmin()]
    print(f"\n📏 Best Boundary (Lowest HD95):")
    print(f"   Steps: {int(best_hd_row['Steps'])} | Start_T: {int(best_hd_row['Start_T'])}")
    print(f"   HD95: {best_hd_row['HD95']:.2f} | Dice: {best_hd_row['Dice']:.4f}")
    
    # 3. 最佳效率平衡 (Best Efficient)
    # 逻辑: 找到Dice在最佳Dice的99.5%以内，且Steps最少的组合
    threshold = best_dice_row['Dice'] * 0.995
    candidates = df[df['Dice'] >= threshold].sort_values(by='Steps')
    balanced_row = candidates.iloc[0]
    
    print(f"\n⚡ Best Efficiency (Top 0.5% Performance with Min Steps):")
    print(f"   Steps: {int(balanced_row['Steps'])} | Start_T: {int(balanced_row['Start_T'])}")
    print(f"   Dice: {balanced_row['Dice']:.4f} (vs Best: {best_dice_row['Dice']:.4f})")
    print(f"   Inference Speedup: {int(best_dice_row['Steps']/balanced_row['Steps'])}x faster than best quality")

if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/inference_output/batch_grid_search/1000timesteps"
    output_img = "parameter_heatmap.png"
    
    df = analyze_grid_results(root_dir)
    if not df.empty:
        plot_heatmaps(df, output_img)
        recommend_parameters(df)
    else:
        print("Could not parse data.")