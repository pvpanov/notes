#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit
#%%
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5) * 10  # 5 features
y = X[:, 0] * 3 + X[:, 1] ** 2 - X[:, 2] * 2 + np.random.normal(0, 1, n_samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
#%%
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

lgbm_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[train_data, test_data])

lgb_preds = lgbm_model.predict(X_test, raw_score=True)
#%%
def predict_with_traversal(X, model):
    """Replicate LightGBM regression predictions using tree traversal."""
    tree_data = model.dump_model()["tree_info"]
    predictions = np.zeros(X.shape[0])
    
    for tree in tree_data:
        active_leaf = np.zeros(X.shape[0])
        
        def traverse_tree(node, sample_indices):
            if "leaf_index" in node:
                # Assign leaf value to samples in this leaf
                for idx in sample_indices:
                    active_leaf[idx] = node["leaf_value"]
            else:
                # Decision node
                feature = node["split_feature"]
                threshold = node["threshold"]
                decision_type = node["decision_type"]

                # Determine split condition
                if decision_type == "<=":
                    left_indices = sample_indices[X[sample_indices, feature] <= threshold]
                    right_indices = sample_indices[X[sample_indices, feature] > threshold]
                else:
                    left_indices = sample_indices[X[sample_indices, feature] < threshold]
                    right_indices = sample_indices[X[sample_indices, feature] >= threshold]

                # Recur for left and right child nodes
                traverse_tree(node["left_child"], left_indices)
                traverse_tree(node["right_child"], right_indices)

        # Start traversal for this tree
        traverse_tree(tree["tree_structure"], np.arange(X.shape[0]))
        predictions += active_leaf  # Add contributions from this tree

    return predictions
#%%
# Replicated predictions
custom_preds = predict_with_traversal(X_test, lgbm_model)

# Compare predictions
print("Max Difference:", np.max(np.abs(lgb_preds - custom_preds)))
print("RMSE (LightGBM):", np.sqrt(mean_squared_error(y_test, lgb_preds)))
print("RMSE (Custom):", np.sqrt(mean_squared_error(y_test, custom_preds)))

#%%
@njit(cache=True, nogil=True)
def calculate_crossing_distances(x, threshold, decision_type):
    n_samples = len(x)
    above_distances = []
    below_distances = []
    
    if decision_type == 0:
        below = x <= threshold
    else:
        below = x < threshold
    above = ~below
    upward_crossings = np.where((below[:-1] & above[1:]))[0]
    downward_crossings = np.where((above[:-1] & below[1:]))[0]
    
    for idx in upward_crossings:
        start_idx = idx
        end_idx = idx + 1
        while end_idx < n_samples and x[end_idx] > threshold:
            end_idx += 1
        distance = np.sum(x[start_idx:end_idx] - threshold)
        above_distances.append(distance)
    
    for idx in downward_crossings:
        start_idx = idx
        end_idx = idx + 1
        while end_idx < n_samples and x[end_idx] < threshold:
            end_idx += 1
        distance = np.sum(threshold - x[start_idx:end_idx])
        below_distances.append(distance)
    
    avg_distance_below = 0.
    avg_distance_above = 0.
    if len(above_distances) > 0:
        avg_distance_above = np.mean(np.array(above_distances, dtype=np.float64))
    if len(below_distances) > 0:
        avg_distance_below = np.mean(np.array(below_distances, dtype=np.float64))
        
    return threshold - avg_distance_below, avg_distance_above - threshold

def prepare_tree_paths(tree, X, mult):
    paths = []
    stack = [(tree["tree_structure"], np.zeros((0, 6)))]
    
    while stack:
        node, conditions = stack.pop()
        if "leaf_index" in node:
            paths.append((node["leaf_value"], conditions))
        else:
            feature = node["split_feature"]
            threshold = node["threshold"]
            decision_type = 0 if node["decision_type"] == "<=" else 1
            tdown, tup = calculate_crossing_distances(X[:, feature], threshold, decision_type)
            l = [feature, threshold - tdown * mult, threshold, threshold + tup * mult, decision_type, 1]
            stack.append((node["right_child"], np.vstack([conditions, l])))
            l[-1] = 0
            stack.append((node["left_child"], np.vstack([conditions, l])))
    
    return paths

@njit(cache=True, nogil=True)
def predict_with_paths_numba(X, paths):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float64)
    
    for leaf_value, conditions in paths:
        path_indicator = np.ones(n_samples, dtype=np.float64)
        
        for ftr, thrd, thr, thru, dec_type, direction in conditions:
            dl = thr - thrd
            du = thru - thr
            
            if dl > 0:
                multl = .5 / dl
                offsetl = (thr - .5 * thrd) / dl
            else:
                multl = np.nan
                offsetl = np.nan
                
            if du > 0:
                multu = .5 / du
                offsetu = (thr - .5 * thru) / du
            else:
                multu = np.nan
                offsetu = np.nan
            
            x_ = X[:, int(ftr)]
            for j, (_x, cur_pi) in enumerate(zip(x_, path_indicator)):
                if cur_pi == 0.:
                    continue 
                if dec_type == 0:
                    if direction == 0:
                        if _x <= thr:
                            if _x > thrd:
                                path_indicator[j] = cur_pi * (offsetl - _x * multl)
                        else: 
                            path_indicator[j] = 0.                            
                    else:
                        if _x > thr:
                            if _x < thru:
                                path_indicator[j] = cur_pi * (_x * multu - offsetu)
                        else: 
                            path_indicator[j] = 0.
                else:
                    if direction == 0:
                        if _x < thr:
                            if _x > thrd:
                                path_indicator[j] = cur_pi * (offsetl - _x * multl)
                        else: 
                            path_indicator[j] = 0.                            
                    else:
                        if _x >= thr:
                            if _x < thru:
                                path_indicator[j] = cur_pi * (_x * multu - offsetu)
                        else: 
                            path_indicator[j] = 0.
                path_indicator
        
        predictions += leaf_value * path_indicator
    
    return predictions

def predict_with_sum_of_indicators_fast(X, model, mult):
    model = lgbm_model
    tree_data = model.dump_model()["tree_info"]
    all_paths = []    
    for tree in tree_data:
        tree_paths = prepare_tree_paths(tree, X, mult)
        all_paths.extend(tree_paths)
    return predict_with_paths_numba(X, all_paths)


mults = [0, 2e-5, 4e-5, 8e-5, 16e-5, 32e-5, 64e-5]
dists = []
for mult in mults:
    # dists.append(np.linalg.norm(lgb_preds - predict_with_sum_of_indicators_fast(X_test, lgbm_model, mult)))
    dists.append(np.std(predict_with_sum_of_indicators_fast(X_test, lgbm_model, mult)))
plt.plot(mults, dists)
plt.xscale('log')

#%%
tree_data = prepare_tree_paths(lgbm_model.dump_model()["tree_info"][0], X_test, 0.)
#%%

@njit
def fun():
    return 1 / 0
#%%
import numpy as np
from multiprocessing import shared_memory
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

# Suppose we have large data arrays
X = np.random.randn(1_000, 10)  # large
y = np.random.randn(1_000)

# Create shared memory blocks for X and y
X_shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shm.buf)
X_shared[:] = X  # copy once into shared memory

y_shm = shared_memory.SharedMemory(create=True, size=y.nbytes)
y_shared = np.ndarray(y.shape, dtype=y.dtype, buffer=y_shm.buf)
y_shared[:] = y

def fit_ridge(shared_name_X, shape_X, dtype_X,
              shared_name_y, shape_y, dtype_y,
              alpha):
    """Function to be run in parallel. Attaches to shared mem blocks."""
    # Attach to X
    existing_shm_x = shared_memory.SharedMemory(name=shared_name_X)
    X_view = np.ndarray(shape_X, dtype=dtype_X, buffer=existing_shm_x.buf)
    
    # Attach to y
    existing_shm_y = shared_memory.SharedMemory(name=shared_name_y)
    y_view = np.ndarray(shape_y, dtype=dtype_y, buffer=existing_shm_y.buf)
    
    model = Ridge(alpha=alpha)
    model.fit(X_view, y_view)
    
    # Child does *not* destroy the shared memory,
    # just closes its view.
    existing_shm_x.close()
    existing_shm_y.close()
    return model

alphas = [0.1, 1.0, 10.0]

models = Parallel(n_jobs=3)(
    delayed(fit_ridge)(
        X_shm.name, X.shape, X.dtype, 
        y_shm.name, y.shape, y.dtype, 
        alpha
    )
    for alpha in alphas
)

# Clean up shared memory in parent once done
X_shm.close()
y_shm.close()
X_shm.unlink()
y_shm.unlink()

print("Trained models:", models)
