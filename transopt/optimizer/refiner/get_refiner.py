from transopt.agent.registry import space_refiner_registry



def get_refiner(refiner_name, **kwargs):
    """Create the optimizer object."""
    refiner_class = space_refiner_registry.get(refiner_name)
    config = kwargs

    if refiner_class is not None:
        refiner = refiner_class(config=config)
    else:
        print(f"Refiner '{refiner_name}' not found in the registry.")
        raise NameError
    return refiner