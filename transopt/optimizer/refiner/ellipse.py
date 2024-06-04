from transopt.optimizer.refiner.refiner_base import RefinerBase
from transopt.agent.registry import space_refiner_registry

@space_refiner_registry.register("ellipse")
class EllipseRefiner(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)