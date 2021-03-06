from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%


    y = generalized_logistic(X, L, U, G)
    # z has to be a scaler and a function of y, mean() will do it
    z = y.mean()

    dzdy, = torch.autograd.grad(z, y)  #

    # Then you start doing eq(1)(2)...(6)
    t = X.size(0)  # number of samples - 48
    n = X.size(1)  # number of features in X .. 2


    # numerical_dzdx
    numerical_dzdx = torch.zeros(t, n)
    for i in range(n):
        mask = torch.zeros(n)
        mask[i] = 1
        numerical_dzdx[:, i] = (dzdy * (
                    (generalized_logistic(X + DELTA * mask, L, U, G) - generalized_logistic(X - DELTA * mask, L, U, G)) / (
                        2 * DELTA))).sum(dim=1)

    # numerical_dzdl
    numerical_dzdl = (dzdy * (
                (generalized_logistic(X, L + DELTA, U, G) - generalized_logistic(X, L - DELTA, U, G)) / (
                2 * DELTA))).sum()

    # numerical_dzdu
    numerical_dzdu = (dzdy * (
            (generalized_logistic(X, L, U + DELTA, G) - generalized_logistic(X, L, U - DELTA, G)) / (
            2 * DELTA))).sum()

    # numerical_dzdg
    numerical_dzdg = (dzdy * (
            (generalized_logistic(X, L, U, G + DELTA) - generalized_logistic(X, L, U, G - DELTA)) / (
            2 * DELTA))).sum()

    z.backward()
    # analytical gradients
    dzdx, dzdl, dzdu, dzdg = X.grad, L.grad, U.grad, G.grad

    # Finally check if analytical and numerical gradients match

    def helper(numerical_value, value, tol):
        err = (numerical_value - value).abs().max()
        if err > tol:
            return False, err
        else:
            return True, err

    # test forward
    tanh_correct, tanh_err = helper(generalized_logistic(X, -1, 1, 2), torch.tanh(X), TOL1)
    # test backward
    dzdx_correct, dzdx_err = helper(numerical_dzdx, dzdx, TOL2)
    dzdl_correct, dzdl_err = helper(numerical_dzdl, dzdl, TOL2)
    dzdu_correct, dzdu_err = helper(numerical_dzdu, dzdu, TOL2)
    dzdg_correct, dzdg_err = helper(numerical_dzdg, dzdg, TOL2)

    is_correct = tanh_correct and dzdx_correct and dzdl_correct and dzdu_correct and dzdg_correct and \
                 torch.autograd.gradcheck(generalized_logistic, (X, L, U, G), eps=DELTA, atol=TOL2)



    err = {'dzdx': dzdx_err, 'dzdl': dzdl_err, 'dzdu': dzdu_err, 'dzdg': dzdg_err, 'y': tanh_err}
    torch.save([is_correct, err], 'generalized_logistic_test_results.pt')
    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)
