from transopt.agent.registry import sampler_registry



def get_sampler(sampler_name, **kwargs):
    """Create the optimizer object."""
    sampler_class = sampler_registry.get(sampler_name)

    if sampler_class is not None:
        sampler = sampler_class(config=kwargs)
    else:
        print(f"Sampler '{sampler_name}' not found in the registry.")
        raise NameError
    return sampler