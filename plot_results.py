import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    root_dir = '/root/autodl-tmp/inference_output/batch_grid_search/1000timesteps'
    output_plot_path = '/root/CPUNet/diffusion_refinement_results.png'
    metrics_of_interest = ['Dice', 'HD95', 'MCC', 'IoU']

    data = []

    print(f"Reading results from: {root_dir}")

    # 遍历目录
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found.")
        return

    for dirname in os.listdir(root_dir):
        dirpath = os.path.join(root_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        
        # 解析文件夹名获取参数 "steps10_start100"
        match = re.match(r'steps(\d+)_start(\d+)', dirname)
        if not match:
            continue
        
        steps = int(match.group(1))
        start_t = int(match.group(2))
        
        # 查找结果 txt 文件
        txt_files = glob.glob(os.path.join(dirpath, '*_results.txt'))
        if not txt_files:
            continue
        
        filepath = txt_files[0]
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        row = {'steps': steps, 'start_t': start_t}
        
        # 提取 Refined Results
        try:
            refined_section = content.split('Refined Results:')[1].split('Coarse Results:')[0]
            for m in metrics_of_interest:
                val = re.search(f'{m}: ([\d\.]+)', refined_section)
                if val:
                    row[f'{m}'] = float(val.group(1))
            
            # 提取 Coarse Results (用作基准线)
            coarse_section = content.split('Coarse Results:')[1]
            for m in metrics_of_interest:
                val = re.search(f'{m}: ([\d\.]+)', coarse_section)
                if val:
                    row[f'{m}_coarse'] = float(val.group(1))
            
            data.append(row)
        except IndexError:
            print(f"Warning: Could not parse file {filepath}")
            continue

    if not data:
        print("No data found!")
        return

    df = pd.DataFrame(data)
    
    # 排序
    df = df.sort_values(by=['steps', 'start_t'])

    print(f"Found {len(df)} data points.")
    print("Generating plots...")

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    unique_steps = sorted(df['steps'].unique())

    for idx, metric in enumerate(metrics_of_interest):
        ax = axes[idx]
        
        # 绘制 Refined 曲线
        # x轴: start_t, y轴: metric, hue: steps
        sns.lineplot(
            data=df, 
            x='start_t', 
            y=metric, 
            hue='steps', 
            style='steps',
            markers=True, 
            dashes=False,
            palette='viridis', 
            ax=ax,
            linewidth=2,
            markersize=8
        )
        
        # 绘制 Coarse Baseline (虚线)
        # 理论上 Coarse 结果应该是一样的，我们取平均值
        baseline_val = df[f'{metric}_coarse'].mean()
        ax.axhline(baseline_val, color='r', linestyle='--', linewidth=2, label=f'Coarse Baseline ({baseline_val:.4f})')
        
        # 为了让 HD95 看起来更符合直觉（越低越好），我们不需要反转坐标轴，但要注意解读
        # 图表标题和标签
        ax.set_title(f'{metric} vs Start Timestep', fontsize=14)
        ax.set_xlabel('Start Timestep', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(title='Inference Steps')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Diffusion Refinement Performance Analysis', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_plot_path, dpi=300)
    print(f"Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
