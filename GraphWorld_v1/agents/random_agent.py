class RandomAgent():

    def __init__(self, label, node):
        self.label = label
        self.node = node
        self.observation = {"Label": self.label, "Coords" : self.node.return_coords()} 