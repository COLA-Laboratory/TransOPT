import requests
import time


class ExperimentClient:
    def __init__(self, server_url, timeout=10):
        self.server_url = server_url
        self.timeout = timeout

    def _handle_response(self, response):
        if response.status_code != 200:
            raise Exception(
                f"Server returned status code {response.status_code}: {response.text}"
            )
        return response.json()

    def start_experiment(self, params):
        try:
            response = requests.post(
                f"{self.server_url}/start_experiment", json=params, timeout=self.timeout
            )
            data = self._handle_response(response)
            return data.get("task_id")
        except requests.RequestException as e:
            raise Exception(f"Failed to start experiment: {e}")

    def get_experiment_result(self, task_id):
        try:
            response = requests.get(
                f"{self.server_url}/get_experiment_result/{task_id}",
                timeout=self.timeout,
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            raise Exception(
                f"Failed to get experiment result for task ID {task_id}: {e}"
            )

    def wait_for_result(self, task_id, poll_interval=2):
        while True:
            result = self.get_experiment_result(task_id)
            if result["state"] == "SUCCESS":
                return result["result"]
            elif result["state"] == "FAILURE":
                raise Exception(f"Experiment failed with status: {result['status']}")
            else:
                print(f"Experiment state: {result['state']}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    client = ExperimentClient(server_url="http://192.168.3.49:5000")

    params = {"param1": "value1", "param2": "value2"}  # Example parameters

    try:
        task_id = client.start_experiment(params)
        print(f"Experiment started with task ID: {task_id}")

        result = client.wait_for_result(task_id)
        print(f"Experiment result: {result}")
    except Exception as e:
        print(f"Error: {e}")