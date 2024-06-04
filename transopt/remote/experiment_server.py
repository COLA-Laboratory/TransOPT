from flask import Flask, jsonify, request
from transopt.remote import celery_inst, ExperimentTaskHandler


class ExperimentServer:
    def __init__(self, task_handler):
        self.app = Flask(__name__)
        self.task_handler = task_handler
        self._setup_routes()

    def _validate_params(self, params):
        required_keys = ["benchmark", "id", "budget", "seed", "bench_params", "fitness_params"]
        return all(key in params for key in required_keys)

    def _setup_routes(self):
        @self.app.route("/start_experiment", methods=["POST"])
        def start_experiment():
            params = request.json

            if not self._validate_params(params):
                return jsonify({"error": "Invalid parameters"}), 400

            try:
                task = self.task_handler.start_experiment(params)
                return jsonify({"task_id": task.id}), 200
            except Exception as e:
                # TODO:
                #   - better error handling
                return jsonify({"error": str(e)}), 500

        

        @self.app.route("/get_experiment_result/<task_id>", methods=["GET"])
        def get_experiment_result(task_id):
            task = celery_inst.AsyncResult(task_id)
            if task.state == "PENDING":
                response = {
                    "state": task.state,
                    "status": "Task is pending...",
                }
            elif task.state != "FAILURE":
                response = {
                    "state": task.state,
                    "result": task.result,
                }
            else:
                # task failed
                response = {
                    "state": task.state,
                    "status": str(task.info),  # this is the exception raised
                }
            return jsonify(response)

    def run(self, host="0.0.0.0", port=5000):
        self.app.run(host=host, port=port)


if __name__ == "__main__":
    task_handler = ExperimentTaskHandler()
    server = ExperimentServer(task_handler=task_handler)
    server.run()