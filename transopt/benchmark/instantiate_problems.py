from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.tab_problem import TabularProblem
from transopt.benchmark.problem_base.transfer_problem import TransferProblem, RemoteTransferOptBenchmark


def InstantiateProblems(
    tasks: dict = None, seed: int = 0, remote: bool = False, server_url: str = None
) -> TransferProblem:
    tasks = tasks or {}

    if remote:
        if server_url is None:
            raise ValueError("Server URL must be provided for remote testing.")
        transfer_problems = RemoteTransferOptBenchmark(server_url, seed)
    else:
        transfer_problems = TransferProblem(seed)

    for task_name, task_params in tasks.items():
        budget = task_params.get("budget", 0)
        workloads = task_params.get("workloads", [])
        budget_type = task_params.get("budget_type", 'Num_FEs')
        params = task_params.get("params", {})


        problem_cls = problem_registry[task_name]
        if problem_cls is None:
            raise KeyError(f"Task '{task_name}' not found in the problem registry.")

        for idx, workload in enumerate(workloads):
            problem = problem_cls(
                task_name=f"{task_name}",
                task_id=idx,
                budget_type=budget_type,
                budget=budget,
                seed=seed,
                workload=workload,
                params=params,
            )
            transfer_problems.add_task(problem)

    return transfer_problems
