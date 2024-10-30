import numpy as np
from transopt.benchmark.HPO.HPO import HPO_ERM
import random

def random_search(n_trials, task_name, budget_type, budget, seed, workload):
    hpo = HPO_ERM(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, optimizer='random')
    
    original_ranges = hpo.configuration_space.original_ranges
    n_var = len(original_ranges)
    xl = np.array([original_ranges[key][0] for key in original_ranges])
    xu = np.array([original_ranges[key][1] for key in original_ranges])
    
    # 用于存储已经尝试过的配置
    tried_configs = set()
    
    best_val_acc = 0
    best_config = None
    
    for trial in range(n_trials):
        # 生成新的配置，直到得到一个未尝试过的配置
        while True:
            config = {}
            for i, name in enumerate(original_ranges.keys()):
                config[name] = np.random.uniform(xl[i], xu[i])
            
            # 将配置转换为不可变的类型（元组），以便可以添加到集合中
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in tried_configs:
                tried_configs.add(config_tuple)
                break
        
        # 设置固定的fidelity值
        
        # 运行目标函数
        result = hpo.objective_function(configuration=config)
        val_acc = 1 - result['function_value']  # 因为我们最小化的是1-accuracy
        
        print(f"Trial {trial + 1}/{n_trials}")
        print(f"Configuration: {config}")
        print(f"Validation Accuracy: {val_acc}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
        
        print(f"Best Validation Accuracy so far: {best_val_acc}")
        print("--------------------")
    
    print("\nRandom Search Completed")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Accuracy: {best_val_acc}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(0)
    random.seed(0)
    
    # 运行随机搜索
    random_search(
        n_trials=5000,  # 指定随机搜索的次数
        task_name='random_search_hpo',
        budget_type='FEs',
        budget=5000,
        seed=0,
        workload=0  # 对应于 RobCifar10 数据集
    )
