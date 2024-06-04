import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE."

from benchmark.synthetic import synthetic_problems
from transopt.utils.serialization import ndarray_to_vectors, vectors_to_ndarray
from transopt.utils.Normalization import normalize


def visual_contour(
    optimizer,
    testsuites,
    train_x,
    train_y,
    Ac_candi,
    test_size=101,
    ac_model=None,
    dtype=np.float64,
):
    # Initialize plots
    f, ax = plt.subplots(2, 2, figsize=(16, 16))

    search_space_info = optimizer.get_spaceinfo("search")

    var_name = [var["name"] for var in search_space_info]
    search_bound = [(var["domain"][0], var["domain"][1]) for var in search_space_info]

    # optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(
        np.linspace(search_bound[0][0], search_bound[0][1], test_size, dtype=dtype),
        np.linspace(search_bound[1][0], search_bound[1][1], test_size, dtype=dtype),
    )
    test_x = np.concatenate(
        (
            xgrid_0.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
            xgrid_1.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
        ),
        axis=1,
    )

    test_vec = ndarray_to_vectors(var_name, test_x)

    observed_pred_y, observed_corv = optimizer.predict(test_x)
    observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
    observed_corv = observed_corv.reshape(xgrid_0.shape)

    # Calculate the true value
    test_x_design = [optimizer._to_designspace(v) for v in test_vec]
    testsuites.lock()
    test_y = testsuites.f(test_x_design)
    test_y = [y["function_value"] for y in test_y]

    mean = np.mean(train_y)
    std = np.std(train_y)
    test_y = normalize(test_y, mean, std)
    test_y = np.array(test_y).reshape(xgrid_0.shape)

    # Calculate EI for the problem
    if ac_model is not None:
        test_ei = ac_model._compute_acq(test_x)
        test_ei = test_ei.reshape(xgrid_0.shape)

    candidate = optimizer._to_searchspace(Ac_candi[0])
    candidate = [v for x, v in candidate.items()]

    def ax_plot(title, ax, train_x, plot_y, test_size, cmap):
        ax.plot(train_x[:, 0], train_x[:, 1], "k*")
        # Predictive mean as blue line
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h1 = ax.contourf(
                xgrid_0,
                xgrid_1,
                plot_y,
                np.arange(-3, 3.5, 0.5),
                cmap=cmap,
            )
            c1 = plt.colorbar(h1, ax=ax)
            # ax.clabel(C, inline=True)
            min_loc_1 = (
                int(np.argmin(plot_y) / test_size),
                np.remainder(np.argmin(plot_y), test_size),
            )
            ax.plot(xgrid_0[min_loc_1], xgrid_1[min_loc_1], "b*")

        ax.set_xlim([-1, 1])
        ax.set_title(title)

    # PLot true contour in the left plot
    ax_plot(
        "iter_" + str(train_x.shape[0]),
        ax[0][0],
        train_x,
        test_y.reshape(xgrid_0.shape),
        test_size,
        cm.Reds,
    )

    ax_plot(
        "Prediction",
        ax[0][1],
        train_x,
        observed_pred_y.reshape(xgrid_0.shape),
        test_size,
        cm.Blues,
    )

    def ax_plot_ei(title, ax, train_x, plot_ei, candidate, cmap):
        # Predictive mean as blue line
        h1 = ax.contourf(xgrid_0, xgrid_1, plot_ei, np.arange(-3, 3.5, 0.5), cmap=cmap)
        c1 = plt.colorbar(h1, ax=ax)
        max_loc = (
            int(np.argmax(plot_ei) / test_size),
            np.remainder(np.argmax(plot_ei), test_size),
        )
        ax.plot(xgrid_0[max_loc], xgrid_1[max_loc], "g*")
        ax.plot(candidate[0], candidate[1], color="orange", marker="*", linewidth=0)
        ax.set_title(title)

    if ac_model is not None:
        ax_plot_ei(
            "Acquisition Function", ax[1][1], train_x, test_ei, candidate, cm.Greens
        )

    # PLot covariance contour in the last row
    ax_plot(
        "Prediction covariance", ax[1][0], train_x, observed_corv, test_size, cm.Blues
    )

    plt.grid()

    Exper_folder = optimizer.exp_path
    if not os.path.exists(
        "{}/verbose/contour/{}/{}".format(
            Exper_folder, optimizer.optimizer_name, f"{testsuites.get_curname()}"
        )
    ):
        os.makedirs(
            "{}/verbose/contour/{}/{}".format(
                Exper_folder, optimizer.optimizer_name, f"{testsuites.get_curname()}"
            )
        )

    plt.savefig(
        "{}/verbose/contour/{}/{}/{}.png".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"iter_{testsuites.get_query_num()}",
        ),
        format="png",
    )
    plt.close()
    testsuites.unlock()


def visual_oned(
    optimizer, testsuites, train_x, train_y, Ac_candi, ac_model=None, dtype=np.float64
):
    # Initialize plots
    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    search_space_info = optimizer.get_spaceinfo("search")

    var_name = [var["name"] for var in search_space_info]
    search_bound = [
        search_space_info[0]["domain"][0],
        search_space_info[0]["domain"][1],
    ]
    test_x = np.arange(search_bound[0], search_bound[1] + 0.005, 0.005, dtype=dtype)

    observed_pred_y, observed_corv = optimizer.predict(test_x[:, np.newaxis])
    test_vec = ndarray_to_vectors(var_name, test_x[:, np.newaxis])
    # Calculate the true value
    test_x_design = [optimizer._to_designspace(v) for v in test_vec]
    testsuites.lock()
    test_y = testsuites.f(test_x_design)
    test_y = np.array([y["function_value"] for y in test_y])

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    test_y = normalize(test_y, y_mean, y_std)
    train_y_temp = normalize(train_y, y_mean, y_std)

    # Calculate EI for the problem
    if ac_model is not None:
        test_ei = ac_model._compute_acq(test_x[:, np.newaxis])

    pre_mean = observed_pred_y
    pre_best_y = np.min(pre_mean)
    pre_best_x = test_x[np.argmin(pre_mean)]
    pre_up = observed_pred_y + observed_corv
    pre_low = observed_pred_y - observed_corv

    ax.plot(test_x, test_y, "r-", linewidth=1, alpha=1)
    ax.plot(test_x, pre_mean[:, 0], "b-", linewidth=1, alpha=1)
    if ac_model is not None:
        ax.plot(test_x, test_ei[:, 0], "g-", linewidth=1, alpha=1)

    candidate = optimizer._to_searchspace(Ac_candi[0])
    ax.plot(train_x[:, 0], train_y_temp[:, 0], marker="*", color="black", linewidth=0)
    ax.plot(candidate[var_name[0]], 0, marker="*", color="orange", linewidth=0)
    ax.plot(pre_best_x, pre_best_y, marker="*", color="blue", linewidth=0)
    ax.fill_between(test_x, pre_up[:, 0], pre_low[:, 0], alpha=0.2, facecolor="blue")

    Exper_folder = optimizer.exp_path
    if not os.path.exists(
        "{}/verbose/oneD/{}/{}".format(
            Exper_folder, optimizer.optimizer_name, f"{testsuites.get_curname()}"
        )
    ):
        os.makedirs(
            "{}/verbose/oneD/{}/{}".format(
                Exper_folder, optimizer.optimizer_name, f"{testsuites.get_curname()}"
            )
        )

    ax.legend()
    plt.grid()

    plt.savefig(
        "{}/verbose/oneD/{}/{}/{}.png".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"iter_{testsuites.get_query_num()}",
            format="png",
        )
    )

    os.makedirs(
        "{}/verbose/oneD/{}/{}/".format(
            Exper_folder, optimizer.optimizer_name, f"{testsuites.get_curname()}"
        ),
        exist_ok=True,
    )
    np.savetxt(
        "{}/verbose/oneD/{}/{}/{}_true.txt".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"{testsuites.get_query_num()}",
        ),
        np.concatenate((test_x[:, np.newaxis], test_y[:, np.newaxis]), axis=1),
    )
    np.savetxt(
        "{}/verbose/oneD/{}/{}/{}_pred_y.txt".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"{testsuites.get_query_num()}",
        ),
        np.concatenate((test_x[:, np.newaxis], observed_pred_y), axis=1),
    )
    np.savetxt(
        "{}/verbose/oneD/{}/{}/{}_cov_lower.txt".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"{testsuites.get_query_num()}",
        ),
        np.concatenate(
            (test_x[:, np.newaxis], observed_pred_y - observed_corv), axis=1
        ),
    )
    np.savetxt(
        "{}/verbose/oneD/{}/{}/{}_cov_higher.txt".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"{testsuites.get_query_num()}",
        ),
        np.concatenate(
            (test_x[:, np.newaxis], observed_pred_y + observed_corv), axis=1
        ),
    )
    if ac_model is not None:
        np.savetxt(
            "{}/verbose/oneD/{}/{}/{}_ei.txt".format(
                Exper_folder,
                optimizer.optimizer_name,
                f"{testsuites.get_curname()}",
                f"{testsuites.get_query_num()}",
            ),
            np.concatenate((test_x[:, np.newaxis], test_ei), axis=1),
        )

    np.savetxt(
        "{}/verbose/oneD/{}/{}/{}_train.txt".format(
            Exper_folder,
            optimizer.optimizer_name,
            f"{testsuites.get_curname()}",
            f"{testsuites.get_query_num()}",
        ),
        np.concatenate((train_x, train_y_temp), axis=1),
    )

    plt.close()
    testsuites.unlock()


def visual_pf(
    optimizer, testsuites, train_x, train_y, Ac_candi, ac_model=None, dtype=np.float64
):
    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    search_space_info = optimizer.get_spaceinfo("search")

    final_pfront = pareto.find_pareto_only_y(obs_points_dic["ParEGO"])
    pfront_sorted = final_pfront[final_pfront[:, 0].argsort(), :]
    plt.scatter(pfront_sorted[:, 0], pfront_sorted[:, 1], c="r", label="ParEGO")
    plt.vlines(pfront_sorted[0, 0], ymin=pfront_sorted[0, 1], ymax=w_ref[1], colors="r")
    for i in range(pfront_sorted.shape[0] - 1):
        plt.hlines(
            y=pfront_sorted[i, 1],
            xmin=pfront_sorted[i, 0],
            xmax=pfront_sorted[i + 1, 0],
            colors="r",
        )
        plt.vlines(
            x=pfront_sorted[i + 1, 0],
            ymin=pfront_sorted[i + 1, 1],
            ymax=pfront_sorted[i, 1],
            colors="r",
        )
    plt.hlines(
        y=pfront_sorted[-1, 1], xmin=pfront_sorted[-1, 0], xmax=w_ref[0], colors="r"
    )

    var_name = [var["name"] for var in search_space_info]
    search_bound = [
        search_space_info[0]["domain"][0],
        search_space_info[0]["domain"][1],
    ]
    test_x = np.arange(search_bound[0], search_bound[1] + 0.005, 0.005, dtype=dtype)

    observed_pred_y, observed_corv = optimizer.predict(test_x[:, np.newaxis])
    test_vec = ndarray_to_vectors(var_name, test_x[:, np.newaxis])
    # Calculate the true value
    test_x_design = [optimizer._to_designspace(v) for v in test_vec]
    testsuites.lock()
    test_y = testsuites.f(test_x_design)
    test_y = np.array([y["function_value"] for y in test_y])

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    test_y = normalize(test_y, y_mean, y_std)
    train_y_temp = normalize(train_y, y_mean, y_std)
