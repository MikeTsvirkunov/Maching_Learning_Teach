def TruePositive(y_predicted, y_test):
    return sum((y_predicted == True) and (y_test == True))

def TrueNegative(y_predicted, y_test):
    return ((y_predicted == False) and (y_test == False)).sum

def precision(y_predicted, y_test):
    return (y_predicted == y_test).sum()