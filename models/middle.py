from itertools import count


class TensorRecorder:
    def __init__(self, activate=False):
        self.buffer = {}
        self.counter = count(1)
        self.activate = activate

    def record(self, tensor, comment=None):
        self.buffer[next(self.counter)] = {'data': tensor, 'comment': comment}

    def clear(self):
        self.buffer = {}
        self.counter = count(1)
        
    def getState(self):
        return self.activate