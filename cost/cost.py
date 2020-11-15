from algebra import sum_vv
from cost import BasicCost, Regularization, Cost


def make_cost(bc: BasicCost, reg: Regularization) -> Cost:
    def combo(x, theta, y):
        c1, g1 = bc(x, theta, y)
        c2, g2 = reg(theta)
        return c1 + c2, sum_vv(g1, g2)

    return combo
