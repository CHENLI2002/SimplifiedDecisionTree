import graphviz
import main

data = main.get_data("data/D3leaves.txt")
tree = main.make_sub_tree(data)
rules = ""
count = 'Root'

dot = graphviz.Graph()

def visualize_tree(node, graph, parent, label):
    global count
    global rules
    if node.label != None:
        #print(node.label)
        graph.node(count, str(node.label))

        if parent != None:
            graph.edge(parent, count, label)
            count = count + "A"
        else:
            count = count + "A"

    else:
        feature = node.feature
        split = node.split
        graph.node(count, feature)
        name = count

        if parent != None:
            graph.edge(parent, count, label)
            count += "B"
        else:
            count += "B"

        rules += f"{feature} >= {split} or < {split}\n"

        split_label_then = f">= {split}"
        split_label_else = f"< {split}"

        #print(node.children)
        #print(node.label)

        visualize_tree(node.children['then'], graph, name, split_label_then)
        visualize_tree(node.children['else'], graph, name, split_label_else)
    return graph

dot = visualize_tree(tree, dot, None, None)

dot.render(filename="decisionTreeQ4", format='png', cleanup=True)
dot.view(filename="decisionTreeQ4.png")
print(dot.source)
print(rules)



