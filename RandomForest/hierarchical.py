import matplotlib.pyplot as plt, numpy as np, sys
from expression import ExpressionProfile, plot_profiles


class ExpressionHierarchicalCluster (object):
    """Superclass for clustering ExpressionProfiles.
    Subclasses for leaves and inner nodes do most of the work.
    This class has the main method, "cluster". It also provides some printing functionality.
    """

    def pprint(self):
        """My best stab at an ASCII pretty-print of the cluster.
        Inspired by http://stackoverflow.com/questions/4965335/how-to-print-binary-tree-diagram
        """
        # just here for defining the interface; work is done in subclasses
        pass

    def pprint_helper(self, angle, indent):
        """Helper method for pprint; don't call directly.
        Args:
          angle: String, / for a right child and \ for a left child
          indent: String, to print before the node
        """
        # just here for defining the interface; work is done in subclasses
        pass

    def ordered_profiles(self):
        """The profiles in the leaves of the cluster tree, in order from left to right.
        This can be useful for generating the profile plot laid out to correspond to the tree.
        Returns:
          [ ExpressionProfile ]
        """
        l = []
        self.ordered_profiles_helper(l)
        return l


    @staticmethod
    def cluster(eps, linkage='average'):
        """Hierarchically cluster the expression profiles.
        Args:
          eps: [ExpressionProfile], to cluster
          linkage: 'average'/'min'/'max', how to evaluate between-cluster distances
        Returns:
          ExpressionHierarchicalCluster
        """
        # TODO: your code here
        # Start by creating leaves for all the profiles and computing Euclidean distances between each pair.
        nodes = [ExpressionHierarchicalClusterLeaf(ep) for ep in eps]
        distances = {}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dis = sum(k * k for k in nodes[i].ep.values - nodes[j].ep.values) ** 0.5
                distances[(nodes[i], nodes[j])] = dis

        # repeatedly find the closest pair of clusters and merge them into a new inner node that can be used in subsequent iterations.
        # meanwhile Compute cluster-profile and cluster-cluster distances, allowing the choice of average linkage
        while len(nodes) > 1:
            # find the closest pair
            min_dis = min(distances.values())
            node1 = None
            node2 = None
            for k, v in distances.items():
                if v == min_dis:
                    node1 = k[0]
                    node2 = k[1]

            # merge
            merged_node = ExpressionHierarchicalClusterInner(node1, node2)
            nodes.remove(node1)
            nodes.remove(node2)

            # https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
            for k1, k2 in list(distances.keys()):
                if k1 == node1 or k2 == node1 or k1 == node2 or k2 == node2:
                    del distances[(k1, k2)]

            # calculate the new distance form other nodes to the merged node
            for other_node in nodes:
                pair_dis = []
                for ep1 in other_node.ordered_profiles():
                    for ep2 in merged_node.ordered_profiles():
                        pair_dis.append(sum(k * k for k in ep1.values - ep2.values) ** 0.5)
                if linkage == "average":
                    cluster_dis = sum(pair_dis) / len(pair_dis)
                elif linkage == "min":
                    cluster_dis = min(pair_dis)
                else:
                    cluster_dis = max(pair_dis)

                # update the distance dict
                distances[(other_node, merged_node)] = cluster_dis

            # add merged_node
            nodes.append(merged_node)

        return nodes[0]

class ExpressionHierarchicalClusterLeaf (ExpressionHierarchicalCluster):
    """A leaf in a hierarchical cluster, holding a single ExpressionProfile."""
    
    def __init__(self, ep):
        self.ep = ep
        
    def __len__(self):
        return 1

    def __str__(self):
        return str(self.ep)

    def pprint(self):    
        print(str(self))

    def pprint_helper(self, angle, indent):
        print(indent+' '+angle+'-'+str(self))

    def ordered_profiles_helper(self, ops):
        ops.append(self.ep)
    
class ExpressionHierarchicalClusterInner (ExpressionHierarchicalCluster):
    """An inner node in a hierarchical cluster, holding a left and a right child."""
    
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.size = len(self.left) + len(self.right)

    def __len__(self):
        return self.size

    def __str__(self):
        return '('+str(self.left)+', '+str(self.right)+')'

    def pprint(self):
        self.right.pprint_helper('/', ' ')
        print('-*')
        self.left.pprint_helper('\\', ' ')

    def pprint_helper(self, angle, indent):
        self.right.pprint_helper('/', indent + ('   ' if angle=='/' else ' | '))
        print(indent+' '+angle+'-*')
        self.left.pprint_helper('\\', indent + (' | ' if angle=='/' else '   '))

    def ordered_profiles_helper(self, ops):
        self.left.ordered_profiles_helper(ops)
        self.right.ordered_profiles_helper(ops)

# ------------------------------------------------------------------------
# simplistic command-line driver
# python hierarchical.py <filename> <linkage>

if __name__ == '__main__':
    profiles = ExpressionProfile.load(sys.argv[1])
    tree = ExpressionHierarchicalCluster.cluster(profiles, sys.argv[2])
    tree.pprint()
    plot_profiles(tree.ordered_profiles())
    plt.show()
    