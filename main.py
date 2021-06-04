# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
IMAGES_PATH = os.path.join("images", "training_linear_models")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def create_random_data():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def plot_gradient_descent(theta, eta, X_b, X_new, X_new_b, X, y, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


def learning_schedule(t, t0, t1):
    return t0 / (t + t1)


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)   


def plot_model(X_new, X, y, model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

def bgd_path(theta, X, y, l1, l2, core = 1, eta = 0.05, n_iterations = 200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = core * 2/len(X) * X.T.dot(X.dot(theta) - y) + l1 * np.sign(theta) + l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)


def main():


    #**********************************
    #   Linear Regression
    #**********************************

    # create random linear-ish data
    print('Creating linear-ish data...')
    X, y = create_random_data()
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("generated_data_plot")

    # The Normal Equation is a closed form solution to find theta in linear regression model.
    # O(n^3)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # remember that the function used to generate the data was y = 4 + 3X + Gaussian noise. The Normal Eqn found:
    print('Normal Eqn results: {}'.format(theta_best)) # a little off, but close
    # now we can make predictions like below for theta at a given X:
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    print('Prediction at X {} is {}'.format(X_new, y_predict))

    # plot predicted theta
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 2, 0, 15])
    save_fig("linear_model_predictions_plot")


    # Linear regression using scikit SVD decomposition
    # uses pseudoeinverse computed using SVD, more computationally efficient than normal eqn
    # It also handles edge cases better (when matrix may not be invertible)
    # O(n^2)
    print('Fitting Scikit LinearRegression')
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print('lin_reg.intercept_, lin_reg.coef_: {}'.format(lin_reg.intercept_, lin_reg.coef_))
    print('Prediction: {}'.format(lin_reg.predict(X_new)))


    # Linear regression using batch gradient descent
    # Batch Gradient Descent
    # to implement GD, you need to compute grad of cost func wrt each model param theta. This is called the parital derivative.
    # Batch is a poor name, since the batch is the full training set.
    # While still being slow on very large datasets, it is much faster tahn Normal Eqn or SVD decomposition
    theta = np.random.randn(2,1)  # random initialization
    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print('BGD theta: {}'.format(theta))

    # plot for different eta learning rates
    theta_path_bgd = []
    theta = np.random.randn(2,1)  # random initialization
    plt.figure(figsize=(10,4))
    plt.subplot(131); plot_gradient_descent(theta, eta=0.02, X_b=X_b, X_new=X_new, X_new_b=X_new_b, X=X, y=y)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(theta, eta=0.1, X_b=X_b, X_new=X_new, X_new_b=X_new_b, X=X, y=y, theta_path=theta_path_bgd)
    plt.subplot(133); plot_gradient_descent(theta, eta=0.5, X_b=X_b, X_new=X_new, X_new_b=X_new_b, X=X, y=y)
    save_fig("gradient_descent_plot")


    # Stochastic Gradient Descent
    # picks a random instance in the training set and computes gradients based only on that sample
    # It is much less regular than BGD as a result but is faster. It will get close to minimum but never settle on it
    # final param will be good, but not optimal.
    # When cost function is irregular this can actually help avoid local minima, so it has a better chance to find global minimum
    # it is a good idea to reduce the lr over time.
    theta_path_sgd = []
    m = len(X_b)
    np.random.seed(42)
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters
    theta = np.random.randn(2,1)  # random initialization

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:
                y_predict = X_new_b.dot(theta)
                style = "b-" if i > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i, t0, t1)
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)

    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("sgd_plot")

    print('SGD theta: {}'.format(theta))

    # scikit learn SGD implementation
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    print('sgd_reg.intercept_ {}, sgd_reg.coef_ {}'.format(sgd_reg.intercept_, sgd_reg.coef_))


    # Minibatch Gradient Descent
    # A mixture of stochastic and batch - use stochastic mini batches from training set
    # advantage is performnce boost based on hardware optimizations for matrix muls e.g. GPUs
    theta_path_mgd = []
    n_iterations = 50
    minibatch_size = 20
    theta = np.random.randn(2,1)  # random initialization
    t0, t1 = 200, 1000

    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t, t0, t1)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
    print('Mini-batch theta: {}'.format(theta))


    # Comparison of cost minimization
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)
    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    save_fig("gradient_descent_paths_plot")


    #**********************************
    #   Polynomial Regression
    #**********************************

    # Create polynomial (quadratic) data
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_data_plot")

    # Surprsingly you can use a linear model to fit nonlinear data.]
    # A simple way to do this is to add powers of each feature as new features,
    # and then train linear model on this extended feature set.
    # This is known as polynomial regression.
    from sklearn.preprocessing import PolynomialFeatures
    # degree 2 chosen would add all combinations of values up to power 3.
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # X_poly contains original feature X plus the square of the feature.
    # Now fit linear regression model.
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print('lin_reg.intercept_ {}, lin_reg.coef_ {}'.format(lin_reg.intercept_, lin_reg.coef_))
    # test on new datapoints, plot as red line
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_predictions_plot")

    # Notice in this example how too high a degree polynomial will overfit the training data.
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
                ("poly_features", polybig_features),
                ("std_scaler", std_scaler),
                ("lin_reg", lin_reg),
            ])
        polynomial_regression.fit(X, y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("high_degree_polynomials_plot")

    # We knew in this case that the data was quadratic. What would we do if we did not know and want to avoid overfit?
    # Option 1: Cross-validation
    # Option 2: Look at learning curves. plots of performance vs training set size.

    # This is underfitting. As we add samples it doesn't make RMSE much better in training data or validation data.
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])
    save_fig("underfitting_learning_curves_plot")

    # this polynomial regression model does better but slightly overfits the training data (gap between RMSEs)
    polynomial_regression = Pipeline([
            ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
            ("lin_reg", LinearRegression()),
        ])
    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])
    save_fig("learning_curves_plot")


    #**********************************
    #   Regularized Linear Models
    #**********************************
    from sklearn.linear_model import Ridge

    # for linear models, regularization is typically achieved by constraining weights of model.
    # 3 ways to do this: Ridge Regression (Tikhanov regularization), Lasso Regression, Elastic Net

    # Ridge Regression - reg term is added to cost function. Keeps model weights as smal as possible.
    # If alpha is 0, vanilla lin reg. If alpha is very large, all weights are close to zero and result is flat line through mean
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)

    # Different solvers
    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    ridge_reg.fit(X, y)
    print('cholesky solver {}'.format(ridge_reg.predict([[1.5]])))
    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(X, y)
    print('sag solver {}'.format(ridge_reg.predict([[1.5]])))

    # test and plot on linear and polynomial data
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plot_model(X_new, X, y, Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(X_new, X, y, Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
    save_fig("ridge_regression_plot")

    # Ridge regression using SGD. L2 is simply ridge regression
    sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
    sgd_reg.fit(X, y.ravel())
    sgd_reg.predict([[1.5]])


    # Lasso Regression - Least Absolute Shrinkage and Selection Operator Regression
    # Just like Ridge Regression, but uses l1 norm instead of half the l2 norm
    # An important characteristic is that it tends to eliminate the weights of the least important features
    from sklearn.linear_model import Lasso
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plot_model(X_new, X, y, Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(X_new, X, y, Lasso, polynomial=True, alphas=(0, 10**-7, 1), random_state=42)
    save_fig("lasso_regression_plot")

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print('Lasso pred {}'.format(lasso_reg.predict([[1.5]])))


    t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5
    t1s = np.linspace(t1a, t1b, 500)
    t2s = np.linspace(t2a, t2b, 500)
    t1, t2 = np.meshgrid(t1s, t2s)
    T = np.c_[t1.ravel(), t2.ravel()]
    Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
    yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]
    J = (1/len(Xr) * np.sum((T.dot(Xr.T) - yr.T)**2, axis=1)).reshape(t1.shape)
    N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
    N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)
    t_min_idx = np.unravel_index(np.argmin(J), J.shape)
    t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]
    t_init = np.array([[0.25], [-1]])

    # plot the respective loss functions during gradient descent, and the convergence.
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10.1, 8))
    for i, N, l1, l2, title in ((0, N1, 2., 0, "Lasso"), (1, N2, 0,  2., "Ridge")):
        JR = J + l1 * N1 + l2 * 0.5 * N2**2
        
        tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
        t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

        levelsJ=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
        levelsJR=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
        levelsN=np.linspace(0, np.max(N), 10)
        
        path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
        path_JR = bgd_path(t_init, Xr, yr, l1, l2)
        path_N = bgd_path(np.array([[2.0], [0.5]]), Xr, yr, np.sign(l1)/3, np.sign(l2), core=0)

        ax = axes[i, 0]
        ax.grid(True)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.contourf(t1, t2, N / 2., levels=levelsN)
        ax.plot(path_N[:, 0], path_N[:, 1], "y--")
        ax.plot(0, 0, "ys")
        ax.plot(t1_min, t2_min, "ys")
        ax.set_title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16, rotation=0)

        ax = axes[i, 1]
        ax.grid(True)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
        ax.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
        ax.plot(path_N[:, 0], path_N[:, 1], "y--")
        ax.plot(0, 0, "ys")
        ax.plot(t1_min, t2_min, "ys")
        ax.plot(t1r_min, t2r_min, "rs")
        ax.set_title(title, fontsize=16)
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel(r"$\theta_1$", fontsize=16)

    save_fig("lasso_vs_ridge_plot")

    # ElasticNet - Middle ground between the previous two options. Mix of both, with ratio controlled by r.
    # r = 0 is Ridge, r = 1 is Lasso.
    # In general preferred over Lasso because Lasso may behave erratically when num features > num training instances or when features are strongly correlated.
    from sklearn.linear_model import ElasticNet
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X, y)
    print('Elastic Net pred: {}'.format(elastic_net.predict([[1.5]])))


    #**********************************
    #   Logistic Regression Models
    #**********************************
    # print logistic function plot
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    save_fig("logistic_function_plot")

    # IRIS dataset import
    from sklearn import datasets
    iris = datasets.load_iris()
    print(list(iris.keys()))
    print(iris.DESCR)
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

    # fit logistic regression model
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver="lbfgs", random_state=42)
    log_reg.fit(X, y)
    # predict and extract decision boundary to be plotted
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.figure(figsize=(8, 3))
    plt.plot(X[y==0], y[y==0], "bs")
    plt.plot(X[y==1], y[y==1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
    plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    save_fig("logistic_regression_plot")


    # Logistic regression contour plot using decision boundary
    from sklearn.linear_model import LogisticRegression

    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.int)

    log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
    log_reg.fit(X, y)

    x0, x1 = np.meshgrid(
            np.linspace(2.9, 7, 500).reshape(-1, 1),
            np.linspace(0.8, 2.7, 200).reshape(-1, 1),
        )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = log_reg.predict_proba(X_new)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y==0, 0], X[y==0, 1], "bs")
    plt.plot(X[y==1, 0], X[y==1, 1], "g^")

    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


    left_right = np.array([2.9, 7])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    save_fig("logistic_regression_contour_plot")


    # Logistic Regression can be generalized to be mult class - Softmax Regression or Multinomial Log. Reg.
    # A score is computed for each class and then softmax function is applied. The scores are called logits.
    # Cross-entropy is used to measure how well a set of estimated class probabilities match the target class.
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)

    x0, x1 = np.meshgrid(
            np.linspace(0, 8, 500).reshape(-1, 1),
            np.linspace(0, 3.5, 200).reshape(-1, 1),
        )
    X_new = np.c_[x0.ravel(), x1.ravel()]


    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
    plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    save_fig("softmax_regression_contour_plot")



if __name__ == '__main__':
    main()
