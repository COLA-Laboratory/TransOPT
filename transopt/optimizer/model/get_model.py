from transopt.agent.registry import model_registry



def get_model(model_name, **kwargs):
    """Create the optimizer object."""
    model_class = model_registry.get(model_name)
    config = kwargs

    if model_class is not None:
        model = model_class(config=config)
    else:
        print(f"Refiner '{model_name}' not found in the registry.")
        raise NameError
    return model