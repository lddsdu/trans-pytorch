class Config(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def __repr__(self):
        keys = self.keys()
        return " ".join(["{} : {},".format(key, self[key]) for key in keys])

