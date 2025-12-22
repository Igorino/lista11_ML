import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


# Utilidades
def one_hot(y, K):
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1.0
    return Y


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


# MLP simples (1 camada escondida)
class MLP:
    def __init__(self, d, h, K):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.1, (h, K))
        self.b2 = np.zeros(K)

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        Z = H @ self.W2 + self.b2
        P = softmax(Z)
        return H, P

    def loss(self, X, Y):
        _, P = self.forward(X)
        eps = 1e-15
        return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

    def gradients(self, X, Y):
        H, P = self.forward(X)
        n = X.shape[0]

        dZ = (P - Y) / n
        dW2 = H.T @ dZ
        db2 = np.sum(dZ, axis=0)

        dH = (dZ @ self.W2.T) * (1 - H**2)
        dW1 = X.T @ dH
        db1 = np.sum(dH, axis=0)

        return dW1, db1, dW2, db2

    def predict(self, X):
        _, P = self.forward(X)
        return np.argmax(P, axis=1)


# 1) Gradiente Descendente
def train_gd(model, X, Y, lr=0.1, epochs=2000):
    for ep in range(epochs):
        dW1, db1, dW2, db2 = model.gradients(X, Y)
        model.W1 -= lr * dW1
        model.b1 -= lr * db1
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        if (ep + 1) % 500 == 0:
            print(f"[GD] epoch={ep+1} loss={model.loss(X, Y):.6f}")


# 2) Gradiente Conjugado (CG)
def pack_params(model):
    return np.concatenate([
        model.W1.ravel(), model.b1,
        model.W2.ravel(), model.b2
    ])


def unpack_params(theta, model):
    d, h = model.W1.shape
    h2, K = model.W2.shape

    idx = 0
    model.W1 = theta[idx:idx + d*h].reshape(d, h)
    idx += d*h
    model.b1 = theta[idx:idx + h]
    idx += h
    model.W2 = theta[idx:idx + h2*K].reshape(h2, K)
    idx += h2*K
    model.b2 = theta[idx:idx + K]


def train_cg(model, X, Y):
    def f(theta):
        unpack_params(theta, model)
        return model.loss(X, Y)

    def g(theta):
        unpack_params(theta, model)
        dW1, db1, dW2, db2 = model.gradients(X, Y)
        return np.concatenate([
            dW1.ravel(), db1,
            dW2.ravel(), db2
        ])

    theta0 = pack_params(model)
    res = minimize(f, theta0, jac=g, method="CG", options={"maxiter": 200})
    unpack_params(res.x, model)


# 3) Newton aproximado (Gauss–Newton)
def train_newton_approx(model, X, Y, lr=1.0, iters=30):
    for it in range(iters):
        dW1, db1, dW2, db2 = model.gradients(X, Y)

        # Newton "parcial": aplica passo tipo Newton só na saída
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        # Camada escondida segue GD
        model.W1 -= lr * dW1
        model.b1 -= lr * db1

        print(f"[Newton approx] iter={it+1} loss={model.loss(X, Y):.6f}")


# Execução
def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    K = 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = one_hot(y_train, K)

    print("\n=== Gradiente Descendente ===")
    mlp_gd = MLP(d=4, h=10, K=3)
    train_gd(mlp_gd, X_train, Y_train)
    print("Acurácia GD:", np.mean(mlp_gd.predict(X_test) == y_test))

    print("\n=== Gradiente Conjugado ===")
    mlp_cg = MLP(d=4, h=10, K=3)
    train_cg(mlp_cg, X_train, Y_train)
    print("Acurácia CG:", np.mean(mlp_cg.predict(X_test) == y_test))

    print("\n=== Newton Aproximado ===")
    mlp_nt = MLP(d=4, h=10, K=3)
    train_newton_approx(mlp_nt, X_train, Y_train)
    print("Acurácia Newton:", np.mean(mlp_nt.predict(X_test) == y_test))


if __name__ == "__main__":
    main()
