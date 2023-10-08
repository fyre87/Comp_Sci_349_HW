class Node:
    def __init__(self, data):
        self.label = None
        self.children = {}
        self.data = data
        self.is_leaf = False
        self.split_attribute = None #Attribute it was split on
        self.count_correct = 0
        self.count_incorrect = 0
        self.should_prune = False

    def is_pure(self):
        # If 1 or less data points, automatically pure
        if len(self.data) <= 1:
            return True

        value_to_match = self.data[0]['Class'] #All must equal this for the node to be pure
        for i in range(0, len(self.data)):
            if self.data[i]['Class'] != value_to_match:
                # Found a disagreement in the nodes data. Return false
                return False 
        return True