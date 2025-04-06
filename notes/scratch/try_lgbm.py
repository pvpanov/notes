#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
#%%

np.random.seed(42)
m = 12
t = 25
d = 20
w = np.tile(np.array([.5 / (t - 1)] * (t - 1) + [.5]), d)
#%%
m = np.random.randn(len(w)) * np.sqrt(w)
#%%
import numpy as np
from scipy.optimize import minimize

def objective(p_vars, p0, m, w, c1, c2):
    """
    p_vars: array of length (n-1) corresponding to p[1], p[2], ..., p[n-1]
    p0    : scalar, the fixed value of p[0]
    m, w, c1, c2 : arrays of length n
    """
    # Reconstruct the full p from p0 and the variables p[1..n-1]
    p = np.r_[p0, p_vars]
    
    # dp[i] = p[i+1] - p[i], for i = 0..(n-2)
    dp = p[1:] - p[:-1]
    
    # Compute the objective we want to MAXIMIZE
    # sum( m[i]*p[i] - w[i]*p[i]^2 ) - sum( c1[i]*|dp[i]| + c2[i]*(dp[i])^2 )
    first_sum  = np.sum(m * p_vars - w * (p_vars**2))
    second_sum = np.sum(c1 * np.abs(dp) + c2 * (dp**2))
    val = first_sum - second_sum
    
    # Because 'minimize' will try to minimize this function,
    # we return the negative of val so that maximizing 'val'
    # is equivalent to minimizing '-val'
    return -val

def solve_optimization(p0, m, w, c1, c2):
    """
    Solve the optimization problem:
    
       maximize  sum(m * p - w * p^2)
                 - sum(c1 * abs(dp) + c2 * dp^2)
       subject to p[0] = p0  (fixed)
    
    Parameters
    ----------
    p0 : float
        The fixed scalar value for p[0].
    m, w, c1, c2 : 1D numpy arrays (all of the same length n)
        Coefficients in the objective.
        
    Returns
    -------
    p_opt : 1D numpy array
        The solution for p[0..n-1], where p[0] = p0 and p[1..] are optimized.
    max_value : float
        The maximum value of the objective function.
    """
    n = len(m)
    if not (len(w) == len(c1) == len(c2) == n):
        raise ValueError("All coefficient arrays (m, w, c1, c2) must have the same length.")

    # We only optimize over p[1..n-1], so the dimension is n-1
    # Initialize with zeros (or any other guess you may have)
    p_vars0 = np.zeros(n)
    
    # Use SLSQP (or another method) to solve. 
    # No extra constraints here aside from p[0] being fixed
    res = minimize(
        fun=objective,
        x0=p_vars0,
        args=(p0, m, w, c1, c2),
        method='SLSQP'
    )
    
    # Construct the full optimal solution: 
    p_opt = np.concatenate(([p0], res.x))
    
    # Because we minimized -val, the maximum is -res.fun
    max_value = -res.fun
    
    return p_opt, max_value

#%%
p_optimal_0, _ = solve_optimization(0., m, w, [.05] * len(m), [.05] * len(m))
p_optimal_1, _ = solve_optimization(0., m, w, [.05] * len(m), [.1] * len(m))
p_optimal_2, _ = solve_optimization(0., m, w, [.1] * len(m), [.05] * len(m))
#%%
plt.plot(p_optimal_1 - p_optimal_0)
#%%
plt.scatter(p_optimal_0[1:], (m / (2 * w)))
#%%
plt.plot(p_optimal_0[1: 126])
plt.plot((m / (2 * w))[: 125])
#%%
plt.hist(p_optimal_0, bins=20, alpha=.5, density=True)
plt.hist((m / (2 * w)), bins=20, alpha=.5, density=True)
#%%
np.mean(np.abs(np.diff(m / (2 * w))))
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
#%%

# 1. Generate a regression problem
def generate_data(n_samples=1000, n_features=10, noise=0.1):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    true_weights = np.random.rand(n_features)
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y

# Generate dataset
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%

# 2. Solve with Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)

print(f"Ridge Regression MSE: {ridge_mse:.4f}")

#%%
# 3. Solve with PyTorch Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Initialize neural network
input_dim = X.shape[1]
model = NeuralNet(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
#%%
# Training loop with L2 regularization
lambda_l2 = 0.005
epochs = 500
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Add L2 regularization to the loss
    l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
    loss = loss + lambda_l2 * l2_reg

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
#%%
# Evaluate the model
model.eval()
y_pred_nn = model(X_test_tensor).detach().numpy()
nn_mse = mean_squared_error(y_test, y_pred_nn)

print(f"Neural Network MSE: {nn_mse:.4f}")
#%%
# Compare results
print(f"Ridge Coefficients: {ridge_model.coef_}")
print(f"NN Weights: {model.fc.weight.data.numpy().flatten()}")
#%%
from matplotlib import pyplot as plt
plt.scatter(ridge_model.coef_, model.fc.weight.data.numpy().flatten())