class Polinomizer():

    def __init__(self, data, power):
        self.data = data
        self.power = power

    def polinomize(self):
        return [[e**p for e in l for p in range(self.power+1)] for l in self.data]