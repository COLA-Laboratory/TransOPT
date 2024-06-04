
import numpy as np
import itertools as it

def find_pareto(X, y):
    """
    find pareto set in X and pareto frontier in y

    Paremeters
    ----------
    X : numpy.array
        input data
    y : numpy.array
        output data

    Return
    ------
    pareto_front : numpy.array
        pareto frontier in y
    pareto_set : numpy.array
        pareto set in X
    """
    y_copy = np.copy(y)
    pareto_front = np.zeros((0 ,y.shape[1]))
    pareto_set = np.zeros((0 ,X.shape[1]))
    i = 0
    j = 0
    while i < y_copy.shape[0]:
        y_outi = np.delete(y_copy, i, axis  =0)
        # paretoだったら全部false
        flag = np.all(y_outi <= y_copy[i ,:] ,axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y_copy[i ,:]] ,axis = 0)
            pareto_set = np.append(pareto_set, [X[j ,:]] ,axis = 0)
            i += 1
        else :
            y_copy = np.delete(y_copy, i, axis= 0)
        j += 1
    return pareto_front, pareto_set

def find_pareto_only_y(y):
    """
    obtain only pareto frontier in y

    Parameters
    ----------
    y : numpy.array
        output data

    Returns
    -------
    pareto_front : numpy.array
        pareto frontier in y
    """
    y_copy = np.copy(y)
    pareto_front = np.zeros((0 ,y.shape[1]))
    i = 0

    while i < y_copy.shape[0]:
        y_outi = np.delete(y_copy, i, axis  =0)
        # paretoだったら全部false
        flag = np.all(y_outi <= y_copy[i ,:] ,axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y_copy[i ,:]] ,axis = 0)
            i += 1
        else :
            y_copy = np.delete(y_copy, i, axis= 0)
    return pareto_front


def create_cells(pf, ref, ref_inv=None):
    '''
       从N个帕累托前沿创建被帕累托前沿支配的区域的独立单元格数组（最小化目标）。

       参数
       ----
       pf : numpy array
           帕累托前沿（N \times L）
       ref : numpy array
           界定目标上界的参考点（L）
       ref_inv : numpy array
           界定目标下界的参考点（L）（为方便计算）

       返回
       ----
       lower : numpy array
           帕累托前沿截断区域中M个单元格的下界（M \times L）
       upper : numpy array
           帕累托前沿截断区域中M个单元格的上界（M \times L）
       '''
    N, L = np.shape(pf)

    if ref_inv is None:
        ref_inv = np.min(pf, axis=0)

    if N == 1:
        # 1つの場合そのまま返してよし
        return np.atleast_2d(pf), np.atleast_2d(ref)
    else:
        # refと作る超体積が最も大きいものをpivotとする
        hv = np.prod(pf - ref, axis=1)
        pivot_index = np.argmax(hv)
        pivot = pf[pivot_index]
        # print('pivot :', pivot)

        # pivotはそのままcellになる
        lower = np.atleast_2d(pivot)
        upper = np.atleast_2d(ref)

        # 2^Lの全組み合わせに対して再帰を回す
        for i in it.product(range(2), repeat=L):
            # 全て1のところにはパレートフロンティアはもう無い
            # 全て0のところはシンプルなセルになるので上で既に追加済
            iter_index = np.array(list(i)) == 0
            if (np.sum(iter_index) == 0) or (np.sum(iter_index) == L):
                continue

            # 新しい基準点(pivot座標からiの1が立っているところだけref座標に変換)
            new_ref = pivot.copy()
            new_ref[iter_index] = ref[iter_index]

            # 新しいlower側の基準点(計算の都合上) (下側基準点座標からiの1が立っているところだけpivot座標に変換)
            new_ref_inv = ref_inv.copy()
            new_ref_inv[iter_index] = pivot[iter_index]

            # new_refより全次元で大きいPareto解は残しておく必要あり
            new_pf = pf[(pf < new_ref).all(axis=1), :]
            # new_ref_invに支配されていない点はnew_refとnew_ref_invの作る超直方体に射影する
            new_pf[new_pf < new_ref_inv] = np.tile(new_ref_inv, (new_pf.shape[0], 1))[new_pf < new_ref_inv]

            # 再帰
            if np.size(new_pf) > 0:
                child_lower, child_upper = create_cells(new_pf, new_ref, new_ref_inv)

                lower = np.r_[lower, np.atleast_2d(child_lower)]
                upper = np.r_[upper, np.atleast_2d(child_upper)]

    return lower, upper



def find_pareto_from_posterior(X, mean, y):
    """
    find pareto frontier in predict mean of GPR and pareto set in X

    Parameters
    ----------
    X : numpy.array
        input data
    mean : numpy.array
        predict mean of GPR
    y : numpy.array
        output data

    Returns
    -------
    pareto_front : numpy.array
        pareto frontier in y defined by predict mean
    pareto_set : numpy.array
        pareto set in X
    """
    mean_copy = np.copy(mean)
    pareto_front = np.zeros((0 ,mean.shape[1]))
    pareto_set = np.zeros((0 ,X.shape[1]))
    i = 0
    j = 0
    while i < mean_copy.shape[0]:
        mean_outi = np.delete(mean_copy, i, axis  =0)
        # paretoだったら全部false
        flag = np.all(mean_outi <= mean_copy[i ,:] ,axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y[j ,:]] ,axis = 0)
            pareto_set = np.append(pareto_set, [X[j ,:]] ,axis = 0)
            i += 1
        else :
            mean_copy = np.delete(mean_copy, i, axis= 0)
        j += 1
    return pareto_front, pareto_set




def calc_hypervolume(y, w_ref):
    """
    calculate pareto hypervolume

    Parameters
    ----------
    y : numpy.array
        output data
    w_ref : numpy.array
        reference point for calculating hypervolume

    Returns
    -------
    hypervolume : float
        pareto hypervolume
    """
    hypervolume = 0.0e0
    pareto_front = find_pareto_only_y(y)
    v, w = create_cells(pareto_front, w_ref)

    if v.ndim == 1:
        hypervolume = np.prod(w - v)
    else:
        hypervolume = np.sum(np.prod(w - v, axis=1))
    return hypervolume

