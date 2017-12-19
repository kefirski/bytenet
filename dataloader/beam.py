class Beam:
    def __init__(self, prob=1., data=''):
        self.prob = prob
        self.data = data

    def update(self, prob, token):
        return Beam(prob, self.data + token)

    def __str__(self):
        return 'p = {}, data = {}\n'.format(self.prob, self.data)
