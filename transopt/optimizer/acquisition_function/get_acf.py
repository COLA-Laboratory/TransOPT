
from transopt.agent.registry import acf_registry

def get_acf(acf_name, **kwargs):
    """Create the optimizer object."""
    acf_class = acf_registry.get(acf_name)

    if acf_class is not None:
        acf = acf_class(config=kwargs)
    else:
        print(f"ACF '{acf_name}' not found in the registry.")
        raise NameError
    return acf



# def get_acf(acf_name, model, search_space, config, tabular=False):
#     """Create the optimizer object."""
#     acf_class = get_acf.get(acf_name)
#     acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(search_space)
#     if acf_class is not None:
#         acquisition = acf_class(model=model, optimizer=acquisition_optimizer, space=search_space, config=config)
#     else:
#         # 处理任务名称不在注册表中的情况c
#         print(f"Acquisition '{acf_name}' not found in the registry.")
#         raise NameError

#     return acquisition

