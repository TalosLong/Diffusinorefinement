import subprocess
import os
import time

def run_batch_inference():
    # === 配置参数 ===
    
    # Python 解释器路径 (如果是在 Conda 环境中，可以直接用 'python')
    python_executable = "python"
    
    # 脚本路径
    script_path = "inference_diffusion.py"
    
    # 固定模型路径 (根据您的环境调整，这里使用 inference_diffusion.py 中的默认值或您当前使用的模型)
    # 请确保这些路径指向正确的文件
    cpunet_path = '/root/CPUNet/model/TU_Synapse256/+SGDpolyepochmval0.0001_0.002test_CPUNet_CVC_best.pth'
    diffusion_path = '/root/autodl-tmp/model/diffusion_refiner_Synapse_256_bs8_lr0.0001_timesteps1000_20260121_114806/diffusion_refiner_best.pth'
    
    # 基础输出目录
    base_output_dir = '/root/autodl-tmp/inference_output/batch_grid_search/1000timesteps'
    
    # 数据集配置
    dataset_name = 'Synapse'
    root_path = '/root/autodl-tmp/data/Synapse-CVC/test'
    list_dir = '/root/CPUNet/lists/lists_Synapse-CVC'
    
    # === 实验变量组合 ===
    # 在这里修改您想测试的组合
    
    # 1. 采样步数 (DDIM steps)
    inference_steps_list = [10, 20, 50, 100]
    
    # 2. 起始时间步 (Start Timestep) 
    # 从越大的时间步开始，意味着给粗糙掩码加的噪越多，细化程度可能越大，但也可能偏离原图
    start_timesteps_list = [100, 300, 500, 700, 900]
    
    # =================
    
    total_experiments = len(inference_steps_list) * len(start_timesteps_list)
    current_exp = 0
    
    print(f"开始批量推理实验...")
    print(f"总计实验数量: {total_experiments}")
    print(f"Steps 列表: {inference_steps_list}")
    print(f"Start Timesteps 列表: {start_timesteps_list}")
    print("-" * 50)

    for steps in inference_steps_list:
        for start_t in start_timesteps_list:
            current_exp += 1
            
            # 为该组合创建一个唯一的实验名称/输出文件夹
            exp_name = f"steps{steps}_start{start_t}"
            output_dir = os.path.join(base_output_dir, exp_name)
            
            print(f"\n[实验 {current_exp}/{total_experiments}] 正在运行: Steps={steps}, Start={start_t}")
            print(f"输出目录: {output_dir}")
            
            # 构造命令
            cmd = [
                python_executable, script_path,
                "--cpunet_path", cpunet_path,
                "--diffusion_path", diffusion_path,
                "--num_inference_steps", str(steps),
                "--start_timestep", str(start_t),
                "--output_dir", output_dir,
                "--eta", "0.0",   # 确定性采样
                # "--save_images",  # 如果不想保存图片可注释掉以加快速度
                # "--compare_coarse" # 是否计算粗糙掩码指标
            ]
            
            start_time = time.time()
            
            try:
                # 运行命令
                result = subprocess.run(cmd, check=True, text=True)
                
                duration = time.time() - start_time
                print(f"实验完成，耗时: {duration:.2f}秒")
                
            except subprocess.CalledProcessError as e:
                print(f"实验失败! Steps={steps}, Start={start_t}")
                print(f"错误信息: {e}")
                # 可以选择 continue 继续下一个实验，或者 break 停止
                # continue

    print("\n" + "=" * 50)
    print("所有实验已完成。")
    print(f"结果请查看: {base_output_dir}")

if __name__ == "__main__":
    run_batch_inference()
