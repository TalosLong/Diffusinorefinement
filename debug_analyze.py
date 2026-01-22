import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Pandas 显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def parse_results_file(filepath):
    """解析结果TXT文件读取Metrics"""
    metrics = {}
    current_section = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 更宽松的匹配
        if "Refined Results" in line:
            current_section = "Refined"
            continue
        elif "Coarse Results" in line:
            current_section = "Coarse"
            continue
        
        # 匹配 "Metric: Value"
        if ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
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
    subdirs = glob.glob(os.path.join(root_dir, "steps*_start*"))
    
    print(f"DEBUG: Found {len(subdirs)} subdirectories")

    for subdir in subdirs:
        dirname = os.path.basename(subdir)
        
        # 使用更稳健的正则
        # steps10_start100 -> steps=10, start=100
        steps_match = re.search(r'steps(\d+)', dirname)
        start_match = re.search(r'start(\d+)', dirname)
        
        if not steps_match or not start_match:
            print(f"Skipping {dirname}: Regex match failed")
            continue
            
        steps = int(steps_match.group(1))
        start_t = int(start_match.group(1))
        
        result_files = glob.glob(os.path.join(subdir, "*_results.txt"))
        if not result_files:
            print(f"Skipping {dirname}: No results file found")
            continue
            
        # 拿最新的文件（按名字排序通常日期在最后）
        result_files.sort()
        target_file = result_files[-1]
        
        metrics = parse_results_file(target_file)
        
        # 如果没有找到 Dice，打印一下这个文件的内容看看怎么回事
        if 'Dice' not in metrics:
            print(f"WARNING: No Dice found in {target_file}")
            print(f"Keys found: {list(metrics.keys())}")
        
        row = {
            'Steps': steps,
            'Start_T': start_t,
            'Dice': metrics.get('Dice', 0.0), # 默认为0
            'HD95': metrics.get('HD95', 100.0), # 默认为100
            'Dice_Gain': metrics.get('Dice', 0.0) - metrics.get('Coarse_Dice', 0.0)
        }
        data.append(row)

    df = pd.DataFrame(data)
    # 按照 Start_T 和 Steps 排序
    df = df.sort_values(by=['Start_T', 'Steps'])
    return df

def plot_heatmaps(df, output_path):
    # 确保没有全0情况
    if df['Dice'].sum() == 0:
        print("ERROR: All Dice values are 0! Aborting plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 显式转换数据类型
    df['Dice'] = df['Dice'].astype(float)
    df['HD95'] = df['HD95'].astype(float)
    df['Dice_Gain'] = df['Dice_Gain'].astype(float)

    # Pivot Table
    pivot_dice = df.pivot_table(index="Steps", columns="Start_T", values="Dice")
    pivot_hd95 = df.pivot_table(index="Steps", columns="Start_T", values="HD95")
    pivot_gain = df.pivot_table(index="Steps", columns="Start_T", values="Dice_Gain")
    
    # Sort Index & Columns
    pivot_dice.sort_index(axis=0, inplace=True); pivot_dice.sort_index(axis=1, inplace=True)
    pivot_hd95.sort_index(axis=0, inplace=True); pivot_hd95.sort_index(axis=1, inplace=True)
    pivot_gain.sort_index(axis=0, inplace=True); pivot_gain.sort_index(axis=1, inplace=True)

    print("\nDEBUG: Pivot Table (Dice):")
    print(pivot_dice)

    # Plot
    # 使用 vmin 和 vmax 限制颜色映射范围，避免极端值拉伸色阶
    # 使用更柔和的调色板 (YlGnBu, PuBuGn, RdYlBu)
    
    # 1. Dice Heatmap
    # 自动计算合适的显示范围，排除离群值干扰
    dice_min = df['Dice'].min()
    dice_max = df['Dice'].max()
    
    sns.heatmap(pivot_dice, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0],
               vmin=dice_min, vmax=dice_max)
    axes[0].set_title("Dice Score", fontsize=14)
    axes[0].invert_yaxis()
    
    # 2. HD95 Heatmap
    sns.heatmap(pivot_hd95, annot=True, fmt=".2f", cmap="PuBuGn", ax=axes[1])
    axes[1].set_title("HD95", fontsize=14)
    axes[1].invert_yaxis()
    
    # 3. Dice Improvement Heatmap
    gain_max = df['Dice_Gain'].abs().max()
    sns.heatmap(pivot_gain, annot=True, fmt=".4f", cmap="RdYlBu_r", center=0, ax=axes[2])
    axes[2].set_title("Dice Improvement", fontsize=14)
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plot to {output_path}")

if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/inference_output/batch_grid_search/1000timesteps"
    # 使用一个新的文件名，防止缓存
    output_img = "/root/CPUNet/parameter_heatmap_fixed_v2.png" 
    
    print("Starting analysis...")
    df = analyze_grid_results(root_dir)
    
    print("\nFinal Dataframe Head:")
    print(df.head())
    
    if not df.empty:
        plot_heatmaps(df, output_img)
    else:
        print("Dataframe is empty.")
