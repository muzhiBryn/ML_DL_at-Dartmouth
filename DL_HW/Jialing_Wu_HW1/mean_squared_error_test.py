from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%
    # Yes.DZDY should be 1, and you can calculate in the following way:

    y = mean_squared_error(X1, X2);
    z = y

    # backward propagtion
    dzdy, = torch.autograd.grad(z, y)

    z.backward()
    # analytical gradients
    dzdx1, dzdx2 = X1.grad, X2.grad

    t = X1.size(0)
    n = X1.size(1)

    # numerical_dzdx1
    numerical_dzdx1 = torch.zeros(t, n)
    for i in range(t):
        for j in range(n):
            mask = torch.zeros(t, n)
            mask[i, j] = 1
            numerical_dzdx1[i, j] = (dzdy * (
                        (mean_squared_error(X1 + DELTA * mask, X2) - mean_squared_error(X1 - DELTA * mask, X2)) / (
                            2 * DELTA)))

        # numerical_dzdx2
    numerical_dzdx2 = torch.zeros(t, n)
    for i in range(t):
        for j in range(n):
            mask = torch.zeros(t, n)
            mask[i, j] = 1
            numerical_dzdx2[i, j] = (dzdy * (
                    (mean_squared_error(X1, X2 + DELTA * mask) - mean_squared_error(X1, X2 - DELTA * mask)) / (
                    2 * DELTA)))

    def helper(numerical_value, value):
        err = (numerical_value - value).abs().max()
        if err > TOL:
            return False, err
        else:
            return True, err

    dzdx1_correct, dzdx1_err = helper(numerical_dzdx1, dzdx1)
    dzdx2_correct, dzdx2_err = helper(numerical_dzdx2, dzdx2)

    is_correct = dzdx1_correct and dzdx2_correct and torch.autograd.gradcheck(mean_squared_error, (X1,X2),
                                                                                             eps=DELTA, atol=TOL)

    err = {'dzdx1': dzdx1_err, 'dzdx2': dzdx2_err}
    torch.save([is_correct, err], 'mean_squared_error_test_results.pt')
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
