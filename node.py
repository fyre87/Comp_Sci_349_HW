class Node:
    def __init__(self, data):
        self.label = None
        self.children = {}
        self.data = data
        self.is_leaf = False
        self.split_attribute = None #Attribute it was split on

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


    def find_most_common(self):
        classes = {}
        for i in range(0, len(self.data)):
            if self.data[i]['Class'] not in classes:
                classes[self.data[i]['Class']] = 1
            else:
                classes[self.data[i]['Class']] = classes[self.data[i]['Class']] + 1
        
        # Now loop through the count of each class to find the most common one
        items = classes.items()
        common_class = None # Most common element
        common_value = 0 # Number of occurances of most common element
        for item in items:
            if item[1] > common_value:
                common_class = item[0]
        return common_class