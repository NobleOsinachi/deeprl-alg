class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError


    def save(self, path, epoch):
        raise NotImplementedError

    def load(self, path, epoch):
        raise NotImplementedError
