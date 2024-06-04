from transopt.ResultAnalysis.PlotAnalysis import plot_registry
from transopt.ResultAnalysis.TableAnalysis import table_registry
from transopt.ResultAnalysis.TrackOptimization import track_registry
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
from transopt.ResultAnalysis.AnalysisReport import create_report





def analysis_pipeline(Exper_folder, tasks, methods, seeds, args):
    ab = AnalysisBase(Exper_folder, tasks=tasks,methods= methods,seeds= seeds)
    ab.read_data_from_kb()
    Exper_folder = Exper_folder / 'analysis'
    if args.comparision:
        for plot_name, plot_func in plot_registry.items():
            plot_func(ab, Exper_folder)  # 假设你的度量函数需要额外的参数

        for table_name, table_func in table_registry.items():
            table_func(ab, Exper_folder)  # 假设你的度量函数需要额外的参数

    if args.track:
        pass

    if args.report:
        create_report(Exper_folder)


