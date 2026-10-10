import numpy as np
from estimacion_modelo import preparar_matrices, estimar_mco

def ejecutar_cross_validation_repetida(df, k=5, n_repeticiones=1000):
    """
    Ejecuta Validacion Cruzada K-Fold Repetida y retorna la media y desv. estandar del RMSE
    """
    todos_los_rmse = []

    for rep in range(n_repeticiones):
        np.random.seed(rep)
        indices_cv = np.random.permutation(len(df))
        folds = np.array_split(indices_cv, k)

        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])

            tr_fold = df.iloc[train_idx]
            ts_fold = df.iloc[test_idx]

            X_tr, y_tr = preparar_matrices(tr_fold)
            X_ts, y_ts = preparar_matrices(ts_fold)

            beta_fold = estimar_mco(X_tr, y_tr)
            y_pred_fold = X_ts @ beta_fold

            rmse_k = np.sqrt(np.mean((y_ts - y_pred_fold) ** 2))
            todos_los_rmse.append(rmse_k)

    rmse_promedio = np.mean(todos_los_rmse)
    std_rmse = np.std(todos_los_rmse)

    return rmse_promedio, std_rmse