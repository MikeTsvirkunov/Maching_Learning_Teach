def TruePositive(y_predicted, y_test):
    return (y_predicted == 1)[y_test == True].sum()


def TrueNegative(y_predicted, y_test):
    return (y_predicted == 0)[y_test == False].sum()


def FalseNegative(y_predicted, y_test):
    return (y_predicted == 0)[y_test == True].sum()


def FalsePositive(y_predicted, y_test):
    return (y_predicted == 1)[y_test == False].sum()


def accuracy(y_predicted, y_test):
    return (TrueNegative(y_predicted, y_test) + TruePositive(y_predicted, y_test)) / (FalseNegative(y_predicted, y_test) + FalsePositive(y_predicted, y_test) + TrueNegative(y_predicted, y_test) + TruePositive(y_predicted, y_test))


def specificity(y_predicted, y_test):
    return TrueNegative(y_predicted, y_test) / (TrueNegative(y_predicted, y_test) + FalsePositive(y_predicted, y_test))


def precision(y_predicted, y_test):
    return TruePositive(y_predicted, y_test) / (TruePositive(y_predicted, y_test) + FalsePositive(y_predicted, y_test))


def recall(y_predicted, y_test):
    return TruePositive(y_predicted, y_test) / (TruePositive(y_predicted, y_test) + FalseNegative(y_predicted, y_test))