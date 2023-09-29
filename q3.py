import main

data = main.get_data("data/Druns.txt")
tree = main.make_sub_tree(data)

# Here we chang the main.py to print out splits' infoGain for those whose entropy is zero
# This will be changed back to "ou may skip those candidate splits with zero split information (i.e. the entropy of the split), and continue
# the enumeration" after q3