from visualize import Visualize
from engine.value import Value


def main(visualize: bool = False) -> None:
    x = Value(0.5, label="x")
    w = Value(3.14, label="w")
    b = Value(-2.0, label="b")

    y = (x * w + b).sigmoid()
    y._label = "y"
    y.backward()

    print(x)
    print(w)
    print(b)
    print(y)

    if visualize:
        dot = Visualize.draw(y, rankdir="TB")
        dot.render("graph", view=True)


if __name__ == "__main__":
    main(visualize=True)
