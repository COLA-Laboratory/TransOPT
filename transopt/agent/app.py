import json
import os
from multiprocessing import Process, Manager

from flask import Flask, jsonify, request
from flask_cors import CORS
from services import Services

from transopt.agent.registry import *
from transopt.utils.log import logger

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
manager = Manager()

task_queue = manager.Queue()
result_queue = manager.Queue()
db_lock = manager.Lock()

services = Services(task_queue, result_queue, db_lock)

@app.route("/api/generate-yaml", methods=["POST"])
def generate_yaml():
    # try:
    data = request.json
    user_input = data.get("content", {}).get("text", "")
    response_content = services.chat(user_input)
    return jsonify({"message": response_content}), 200
    # except Exception as e:
    #     logger.error(f"Error in generating YAML: {e}")
    #     return jsonify({"error": str(e)}), 500


@app.route("/api/Dashboard/tasks", methods=["POST"])
def report_send_tasks_information():
    all_info = services.get_experiment_datasets()
    all_tasks_info = []
    for task_name, task_info in all_info:
        info = task_info['additional_config']
        info['problem_name'] = task_name
        all_tasks_info.append(info)
    
    
    return jsonify(all_tasks_info), 200


@app.route("/api/Dashboard/charts", methods=["POST"])
def report_update_charts_data():
    data = request.json
    user_input = data.get("taskname", "")
    charts = services.get_report_charts(user_input)
    return jsonify(charts), 200


@app.route("/api/Dashboard/trajectory", methods=["POST"])
def report_update_trajectory_data():
    data = request.json
    user_input = data.get("taskname", "")
    # trajectory, 数据格式和以前一样 {"TrajectoryData":...}
    charts = services.get_report_traj(user_input)
    return jsonify(charts), 200


@app.route("/api/configuration/select_task", methods=["POST"])
def configuration_recieve_tasks():
    tasks_info = request.json
    # try:
    services.receive_tasks(tasks_info) 
    # except Exception as e:
    #     logger.error(f"Error in searching dataset: {e}")
    #     return jsonify({"error": str(e)}), 500
    
    return {"succeed": True}, 200


@app.route("/api/configuration/select_algorithm", methods=["POST"])
def configuration_recieve_algorithm():
    optimizer_info = request.json
    print(optimizer_info)
    # optimizer_info = {'SpaceRefiner': 'default', 
    #                   'SpaceRefinerParameters': '', 
    #                   'SpaceRefinerDataSelector': 'default', 
    #                   'SpaceRefinerDataSelectorParameters': '', 
    #                   'Sampler': 'default', 
    #                   'SamplerParameters': '', 
    #                   'SamplerInitNum': '11',
    #                   'SamplerDataSelector': 'default', 
    #                   'SamplerDataSelectorParameters': '', 
    #                   'Pretrain': 'default', 
    #                   'PretrainParameters': '', 
    #                   'PretrainDataSelector': 'default', 
    #                   'PretrainDataSelectorParameters': '', 
    #                   'Model': 'default', 
    #                   'ModelParameters': '', 
    #                   'ModelDataSelector': 'default', 
    #                   'ModelDataSelectorParameters': '', 
    #                   'ACF': 'default', 
    #                   'ACFParameters': '', 
    #                   'ACFDataSelector': 'default', 
    #                   'ACFDataSelectorParameters': '', 
    #                   'Normalizer': 'default', 
    #                   'NormalizerParameters': '', 
    #                   'NormalizerDataSelector': 'default', 
    #                   'NormalizerDataSelectorParameters': ''}
    try:
        services.receive_optimizer(optimizer_info)
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500
    
    return {"succeed": True}, 200


@app.route("/api/configuration/basic_information", methods=["POST"])
def configuration_basic_information():
    data = request.json
    user_input = data.get("paremeter", "")

    task_data = services.get_modules()
    # with open('transopt/agent/page_service_data/configuration_basic.json', 'r') as file:
    #     data = json.load(file)
    print(services)
    return jsonify(task_data), 200


@app.route("/api/configuration/dataset", methods=["POST"])
def configuration_dataset():
    metadata_info = request.json
    # print(metadata_info)
    # metadate_info = {
    #     "object": "Space refiner",
    #     "datasets": ["dataset1", "dataset2]
    # }
    if metadata_info['object'] == 'Narrow Search Space':
        metadata_info['object'] = 'SpaceRefiner'
    elif metadata_info['object'] == 'Initialization':
        metadata_info['object'] = 'Sampler'
    elif metadata_info['object'] == 'Pre-train':
        metadata_info['object'] = 'Pretrain'
    elif metadata_info['object'] == 'Surrogate Model':
        metadata_info['object'] = 'Model'
    elif metadata_info['object'] == 'Acquisition Function':
        metadata_info['object'] = 'ACF'
    
    try:
        services.set_metadata(metadata_info)
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500
    
    return {"succeed": True}, 200


@app.route("/api/configuration/search_dataset", methods=["POST"])
def configuration_search_dataset():
    try:
        data = request.json

        dataset_name = data["task_name"]
        if data['search_method'] == 'Fuzzy' or 'Hash':
            dataset_info = {}
        elif data['search_method'] == 'LSH':
            dataset_info = {
                "num_variables": data["num_variables"],
                "num_objectives": data["num_objectives"],
                "variables": [
                    {"name": var_name} for var_name in data["variables_name"].split(",")
                ],
            }
        else:
            pass
        datasets = services.search_dataset(data['search_method'], dataset_name, dataset_info)

        return jsonify(datasets), 200
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/configuration/delete_dataset", methods=["POST"])
def configuration_delete_dataset():
    metadata_info = request.json
    datasets = metadata_info["datasets"]
    services.remove_dataset(datasets) 
    return {"succeed": True}, 200


@app.route("/api/configuration/run", methods=["POST"])
def configuration_run():
    run_info = request.json
    
    seeds = [int(seed) for seed in run_info['Seeds'].split(",")]
    services.run_optimize(seeds)  # Handle process creation within run_optimize
    
    return jsonify({"isSucceed": True}), 200

@app.route("/api/configuration/run_progress", methods=["POST"])
def configuration_run_progress():
    message = request.json
    # 获取正在运行的任务的进度
    data = []
    process_info = services.get_all_process_info()
    for subpross_id, subpross  in process_info.items():
        if subpross['status'] == 'running':
            data.append({
                "name": f"{subpross['task']}_pid_{subpross_id}",
                "progress": str(subpross['progress']),
            })
    
    return jsonify(data), 200

@app.route("/api/configuration/stop_progress", methods=["POST"])
def configuration_stop_progress():
    message = request.json
    task_name = message['name']
    print(task_name)
    pid = int(task_name.split('_')[-1])
    services.terminate_task(pid)

    return {"succeed": True}, 200


@app.route("/api/RunPage/get_info", methods=["POST"])
def run_page_get_info():
    data = request.json
    user_input = data.get("action", "")

    task_data = services.get_configuration()
    with open('transopt/agent/page_service_data/configuration_info.json', 'r') as file:
        data = json.load(file)
    return jsonify(task_data), 200


@app.route("/api/comparison/selections", methods=["POST"])
def comparison_send_selections():
    info = request.json
    # Comparison初始化时，请求可选择的搜索选项
    data = services.get_comparision_modules()

    return jsonify(data), 200


@app.route("/api/comparison/choose_task", methods=["POST"])
def comparison_choose_tasks():
    conditions = request.json
    ret = []
    charts_data = {}
    for condition in conditions:
        ret.append(services.comparision_search(condition)) 
    

    charts_data['BoxData'] = services.get_box_plot_data(ret)
    charts_data['TrajectoryData'] = services.construct_statistic_trajectory_data(ret)
    return jsonify(charts_data), 200


if __name__ == "__main__":
    app.run(debug=services.config.DEBUG)
