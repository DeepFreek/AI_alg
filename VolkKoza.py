VARIABLES = ["W", "C", "G"]
CONSTRAINTS = [
    ["W","G"],
    ["C", "G"]
]
class Node():
    def __init__ (self, left, right, shore):
        self.left = left     
        self.right = right
        self.shore = shore
        
    def __eq__(self, node):
        if self.left == node.left and self.right == node.right and self.shore == node.shore:
            return True
        else:
            return False

def consistent(assignment=[]):
    for i in CONSTRAINTS:
        i.sort()
        if i == assignment:
            return False
    return True
def transfer(node, path):

    node.left.sort()
    node.right.sort()
    

    if node.left == []:
        path.append(node)
        return path
    

    if node.shore == 'left':
        for i in node.left:
            new_node = Node(node.left.copy(), node.right.copy(), node.shore)
            new_node.right.append(i)
            new_node.right.sort()
            new_node.left.remove(i)
            new_node.left.sort()
            new_node.shore = 'right'

            
            if new_node in path:
                if consistent(node.left):
                    path.append(Node(node.left.copy(), node.right.copy(), node.shore))
                    node.shore = 'right'
                    result = transfer(node, path)
                    if result is not None:
                        return result
                else: continue




            elif consistent(new_node.left):
                path.append(Node(node.left.copy(), node.right.copy(), node.shore))
                result = transfer(new_node, path)
                if result is not None:
                    return result
    else:
        for i in node.right:
            new_node = Node(node.left.copy(), node.right.copy(), node.shore) 
            new_node.left.append(i)
            new_node.left.sort()
            new_node.right.remove(i) 
            new_node.right.sort()
            new_node.shore = 'left'
            if new_node in path:
                if consistent(node.right):
                    path.append(Node(node.left.copy(), node.right.copy(), node.shore))
                    node.shore = 'left'
                    result = transfer(node, path)
                    if result is not None:
                        return result
                else: continue
            elif consistent(new_node.right):
                path.append(Node(node.left.copy(), node.right.copy(), node.shore))
                result = transfer(new_node, path)
                if result is not None:
                    return result
    return None
node = Node(VARIABLES, [], 'left')
for j in transfer(node, []):
    print(j.left, j.right, j.shore)