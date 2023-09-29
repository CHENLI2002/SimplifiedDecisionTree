import q4
import main
import graphviz

data_d1 = main.get_data('data/D1.txt')
data_d2 = main.get_data('data/D2.txt')
tree_1 = main.make_sub_tree(data_d1)
tree_2 = main.make_sub_tree(data_d2)
d1_tree = graphviz.Graph()
d2_tree = graphviz.Graph()

d1_tree = q4.visualize_tree(tree_1, d1_tree, None, None)
d2_tree = q4.visualize_tree(tree_2, d2_tree, None, None)

d1_tree.render(filename="D1Q5Tree", format='png', cleanup=True)
d1_tree.view(filename="D1Q5Tree.png")
d2_tree.render(filename="D2Q5Tree", format='png', cleanup=True)
d2_tree.view(filename="D2Q5Tree.png")
