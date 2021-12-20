# TODO: load and fit the current optimal number of states
import json
import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations, permutations
from anytree import Node, RenderTree, Walker, PreOrderIter, findall, findall_by_attr, search
# ------------------------------------------------------------------------------
class data_map:
    def __init__(self,dat):
        # map parameters
        self.N = dat['N']  # total city number, including start
        self.total = dat['total']  # total budget

        self.R = dat['R'] # 200 * 200  # circle radius' sqaure
        self.r = dat['r'] # np.random.uniform(0, self.R, self.N).tolist()
        self.phi = dat['phi'] # np.random.uniform(0, 2 * math.pi, self.N).tolist()
        self.x = dat['x'] # np.sqrt(self.r) * np.cos(self.phi)
        # self.x = self.x.astype(int).tolist()
        self.y = dat['y'] # np.sqrt(self.r) * np.sin(self.phi)
        # self.y = self.y.astype(int).tolist()
        self.xy = dat['xy'] # [(self.x[i], self.y[i]) for i in range(0, len(self.x))]  # combine x and y

        self.city_start = dat['city_start'] # self.xy[0]  # start city
        self.distance = dat['distance'] # distance_matrix(self.xy, self.xy, p=2, threshold=10000)  # city distance matrix
# ------------------------------------------------------------------------------
# breath first search with budget
def optimal(mmap, now):
    all_ = list(range(0, mmap.N))
    for j in now.path:
        all_.remove(j.name)

    for i in all_:
        budget_remain = now.budget - mmap.distance[now.name][i]
        if budget_remain >= 0:
            node = Node(i, parent=now, budget=budget_remain)
            optimal(mmap, node)

class TreeStructure:
    def __init__(self, mmap):
        self.mmap = mmap
        self.nodes_bucket = []

    def get_Tree(self, rootnode = Node(0,budget = 300)):
        self.root_ = rootnode
        self.nodes_bucket.append(self.root_)
        # self.root_ = Node(0)
        self.get_Tree_(self.root_)
        self.curr_node = self.root_

    def get_Tree_(self,now):
        all_ = list(range(0, self.mmap.N))
        for j in now.path:
            all_.remove(j.name)
        bucket_depth = []
        for i in all_:
            # if now.name == 0:
            #     budget_remain = 300        - self.mmap.distance[int(now.name)][i]
            # else:
            budget_remain = now.budget - self.mmap.distance[int(now.name)][i]
            if budget_remain >= 0:
                self.nodes_bucket.append(Node(i, parent=now, budget=budget_remain))
                bucket_depth.append(Node(i,parent=now, budget=budget_remain))
                self.get_Tree_(self.nodes_bucket[-1])

    def render_out(self):
        self.tree = RenderTree(self.root_)
        return print(self.tree)

    def gen_path_flatten(self):
        self.paths = []
        for p in self.nodes_bucket:
            self.paths.append([i.name for i in p.path])

def main():
    # basicMap
    with open('basicMap.json','rb') as f:
        data = json.load(f)

    for i in range(len(data)):
        dat = data[i]
        mmap = data_map(dat)

        TS = TreeStructure(mmap)
        TS.get_Tree()
        rendered_tree = TS.render_out()
        TS.gen_path_flatten()

        leaves = [i.name for i in list(PreOrderIter(TS.root_, filter_=lambda node: node.is_leaf))] # number of leaves
        unique_leaves = np.unique(leaves).tolist()

        path_s = [i.path for i in list(PreOrderIter(TS.root_, filter_=lambda node: node.is_leaf))] # number of paths
        findall_by_attr(TS.root_, 11, maxlevel=2)

if __name__=="__main__":
    main()
