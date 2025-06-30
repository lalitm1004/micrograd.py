from visualize import Visualize
from engine.node import Node


def main(visualize: bool = False) -> None:
    x = Node(0.5, label="x")
    w = Node(3.14, label="w")
    b = Node(-2.0, label="b")

    y = (x * w + b**2).relu()
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
