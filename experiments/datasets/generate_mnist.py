import pandas as pd


def generate_mnist_58(mnist: pd.DataFrame):
    mnist[
        (mnist.label == 5) |
        (mnist.label == 8)
    ].to_csv("mnist_58.csv", index=False)


def generate_mnist_30(mnist: pd.DataFrame):
    mnist.iloc[0:30].to_csv("mnist_30.csv", index=False)


if __name__ == "__main__":
    mnist = pd.read_csv("mnist.csv")
    generate_mnist_58(mnist)
    generate_mnist_30(mnist)
