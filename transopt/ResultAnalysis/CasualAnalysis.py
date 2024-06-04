
from transopt.ResultAnalysis.PlotAnalysis import plot_registry
from transopt.ResultAnalysis.TableAnalysis import table_registry
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
from transopt.ResultAnalysis.AnalysisReport import create_report


def casual_analysis(Exper_folder, tasks, methods, seeds, args):
    ab = AnalysisBase(Exper_folder, tasks=tasks,methods= methods,seeds= seeds)
    ab.read_data_from_kb()
    Exper_folder = Exper_folder / 'analysis'


    
