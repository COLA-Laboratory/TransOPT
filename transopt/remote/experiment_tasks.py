from celery import Celery, Task
from celery.utils.log import get_task_logger
from transopt.agent.registry import problem_registry

celery_inst = Celery(__name__)
celery_inst.config_from_object("celeryconfig")

logger = get_task_logger(__name__)


class DebugTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task [{task_id}] failed: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        logger.warning(f"Task [{task_id}] succeeded with result: {retval}")

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        logger.warning(f"Task [{task_id}] finished with status: {status}")


class ExperimentTaskHandler:
    def __init__(self):
        pass

    @celery_inst.task(bind=True, base=DebugTask)
    def run_experiment(self, params):
        # rdb.set_trace()
        bench_name = params["benchmark"]
        bench_id = params["id"]
        budget = params["budget"]
        seed = params["seed"]
        bench_params = params["bench_params"]
        fitness_params = params["fitness_params"]

        benchmark_cls = problem_registry.get(bench_name)

        if benchmark_cls is None:
            self.update_state(state="FAILURE", meta={"status": "Benchmark not found!"})
            raise ValueError(f"Benchmark {bench_name} not found!")

        try:
            problem = benchmark_cls(
                task_name=f"{bench_name}_{bench_id}",
                task_id=bench_id,
                budget=budget,
                seed=seed,
                params=bench_params,
            )

            result = problem.f(**fitness_params)
            return result
        except Exception as e:
            self.update_state(state="FAILURE", meta={"status": "Experiment failed!"})
            raise e

    def start_experiment(self, params):
        return self.run_experiment.apply_async(args=[params])


if __name__ == "__main__":
    # handler = ExperimentTaskHandler()
    # params = {
    #     "benchmark": "sample_bench",
    #     "id": 1,
    #     "budget": 100,
    #     "seed": 42,
    #     "bench_params": {},
    #     "fitness_params": {}
    # }
    
    # handler.start_experiment(params)
    pass
