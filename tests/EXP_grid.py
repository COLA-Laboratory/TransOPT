import numpy as np
from transopt.benchmark.HPO.HPO import HPO_ERM
from scipy.stats import qmc

def sobol_search(n_samples, task_name, budget_type, budget, seed, workload):
    hpo = HPO_ERM(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, optimizer='sobol')
    original_ranges = hpo.configuration_space.original_ranges
    n_var = len(original_ranges)
    xl = np.array([original_ranges[key][0] for key in original_ranges])
    xu = np.array([original_ranges[key][1] for key in original_ranges])
    
    # 创建Sobol序列采样器
    sampler = qmc.Sobol(d=n_var, scramble=True, seed=seed)
    
    # 生成Sobol序列样本
    sample = sampler.random(n=n_samples)
    
    # 将样本从[0,1]范围映射到参数实际范围
    scaled_sample = qmc.scale(sample, xl, xu)
    
    best_val_acc = 0
    best_config = None
    
    for i in range(n_samples):
        config = {}
        for j, param_name in enumerate(original_ranges.keys()):
            config[param_name] = scaled_sample[i, j]
        
        # 运行目标函数
        result = hpo.objective_function(configuration=config)
        val_acc = 1 - result['function_value']  # 因为我们最小化的是1-accuracy
        
        print(f"Trial {i + 1}/{n_samples}")
        print(f"Configuration: {config}")
        print(f"Validation Accuracy: {val_acc}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
        
        print(f"Best Validation Accuracy so far: {best_val_acc}")
        print("--------------------")
    
    print("\nSobol Search Completed")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Accuracy: {best_val_acc}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(0)
    
    # 运行Sobol序列搜索
    sobol_search(
        n_samples=5000,  # 指定采样数量
        task_name='sobol_search_hpo',
        budget_type='FEs',
        budget=5000,
        seed=0,
        workload=0  # 对应于 RobCifar10 数据集
    )
