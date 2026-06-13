# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:38:53 2020

@author: Harry
"""

def retrieveTree(i):
	listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':\
					{0:'no', 1:'yes'}}}},
				   {'no surfacing':{0:'no',1:{'flippers':\
				    {0:{'head':{0:'no', 1:'yes'}},1:'no'}}}}]
	return listOfTrees[i]
if __name__ == '__main__':
	tree = retrieveTree(1)
	leafs = getNumLeafs(tree)
	depth = getTreeDepth(tree)
	print(leafs)
	print(depth)
