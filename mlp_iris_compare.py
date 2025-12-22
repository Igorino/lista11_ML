import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


# Funções utilitárias

# Converte vetor de rótulos (0,1,2,...) em one-hot
# Ex: y=2, K=3 -> [0,0,1]
def one_hot(y, K):
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1.0
    return Y


# Função softmax estável numericamente
# Converte logits em probabilidades que somam 1
def softmax(Z):
    # subtrai o máximo por linha pra evitar overflow no exp
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


# MLP simples com UMA camada escondida
# Arquitetura:
#   X -> (W1,b1) -> tanh -> (W2,b2) -> softmax
class MLP:
    def __init__(self, d, h, K):
        # gerador aleatório pra inicialização reprodutível
        rng = np.random.default_rng(42)

        # pesos da camada escondida (d -> h)
        self.W1 = rng.normal(0, 0.1, (d, h))
        self.b1 = np.zeros(h)

        # pesos da camada de saída (h -> K)
        self.W2 = rng.normal(0, 0.1, (h, K))
        self.b2 = np.zeros(K)

    def forward(self, X):
        # passagem forward completa

        # ativação da camada escondida
        H = np.tanh(X @ self.W1 + self.b1)

        # logits da saída
        Z = H @ self.W2 + self.b2

        # probabilidades finais
        P = softmax(Z)

        return H, P

    def loss(self, X, Y):
        # calcula a cross-entropy loss
        _, P = self.forward(X)
        eps = 1e-15  # evita o log(0)
        return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

    def gradients(self, X, Y):
        # backpropagation

        H, P = self.forward(X)
        n = X.shape[0]

        # erro na saída (softmax + cross-entropy)
        dZ = (P - Y) / n

        # gradientes da camada de saída
        dW2 = H.T @ dZ
        db2 = np.sum(dZ, axis=0)

        # erro propagado para a camada escondida
        # derivada do tanh = 1 - tanh^2
        dH = (dZ @ self.W2.T) * (1 - H**2)

        # gradientes da camada escondida
        dW1 = X.T @ dH
        db1 = np.sum(dH, axis=0)

        return dW1, db1, dW2, db2

    def predict(self, X):
        # retorna a classe com maior probabilidade
        _, P = self.forward(X)
        return np.argmax(P, axis=1)


# 1) Treinamento com Gradiente Descendente
# Atualiza os parâmetros diretamente com o gradiente
def train_gd(model, X, Y, lr=0.1, epochs=2000):
    for ep in range(epochs):
        # calcula todos os gradientes via backprop
        dW1, db1, dW2, db2 = model.gradients(X, Y)

        # atualiza os parâmetros
        model.W1 -= lr * dW1
        model.b1 -= lr * db1
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        # imprime a loss de tempos em tempos
        if (ep + 1) % 500 == 0:
            print(f"[GD] epoch={ep+1} loss={model.loss(X, Y):.6f}")


# 2) Gradiente Conjugado (CG)
# Usa scipy.optimize.minimize para otimização

# Empacota todos os parâmetros do modelo em um vetor 1D
def pack_params(model):
    return np.concatenate([
        model.W1.ravel(), model.b1,
        model.W2.ravel(), model.b2
    ])


# Desempacota o vetor 1D de volta nos parâmetros do modelo
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
    # função objetivo (loss)
    def f(theta):
        unpack_params(theta, model)
        return model.loss(X, Y)

    # gradiente da função objetivo
    def g(theta):
        unpack_params(theta, model)
        dW1, db1, dW2, db2 = model.gradients(X, Y)
        return np.concatenate([
            dW1.ravel(), db1,
            dW2.ravel(), db2
        ])

    # vetor inicial de parâmetros
    theta0 = pack_params(model)

    # otimização por gradiente conjugado
    res = minimize(
        f, theta0, jac=g, method="CG", options={"maxiter": 200}
    )

    # coloca os parâmetros finais de volta no modelo
    unpack_params(res.x, model)


# 3) Newton Aproximado (Gauss–Newton simplificado)
# Aqui é "Newton" no sentido didático:
# - passo mais agressivo
# - sem Hessiana completa
def train_newton_approx(model, X, Y, lr=1.0, iters=30):
    for it in range(iters):
        # calcula gradientes
        dW1, db1, dW2, db2 = model.gradients(X, Y)

        # "Newton parcial": atualiza a saída com passo maior
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        # camada escondida segue gradiente normal
        model.W1 -= lr * dW1
        model.b1 -= lr * db1

        print(f"[Newton approx] iter={it+1} loss={model.loss(X, Y):.6f}")


# Execução principal
def main():
    # carrega o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    K = 3

    # split treino / teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # normalização dos dados (importante para MLP)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # converte rótulos para one-hot
    Y_train = one_hot(y_train, K)

    # Gradiente Descendente
    print("\n=== Gradiente Descendente ===")
    mlp_gd = MLP(d=4, h=10, K=3)
    train_gd(mlp_gd, X_train, Y_train)
    print("Acurácia GD:", np.mean(mlp_gd.predict(X_test) == y_test))

    # Gradiente Conjugado
    print("\n=== Gradiente Conjugado ===")
    mlp_cg = MLP(d=4, h=10, K=3)
    train_cg(mlp_cg, X_train, Y_train)
    print("Acurácia CG:", np.mean(mlp_cg.predict(X_test) == y_test))

    # Newton (Aproximado, pq o normal é computacionalmente muito complexo)
    print("\n=== Newton Aproximado ===")
    mlp_nt = MLP(d=4, h=10, K=3)
    train_newton_approx(mlp_nt, X_train, Y_train)
    print("Acurácia Newton:", np.mean(mlp_nt.predict(X_test) == y_test))


# ponto de entrada do script
if __name__ == "__main__":
    main()
