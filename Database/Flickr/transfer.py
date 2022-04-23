import sys
import numpy as np

name = sys.argv[1]

feature_size = sys.argv[2]

fp = open('{}.label'.format(name))
wp = open("group.txt",'w')

label_lst=[]

for line in fp:
    label_lst.append(line.strip())

num_nodes = len(label_lst)

for index,value in enumerate(label_lst):
    wp.write("{} {}\n".format(index,value))

fp.close()
wp.close()

fp = open('{}.node'.format(name))
wp = open("features.txt",'w')

z  = np.zeros((num_nodes,int(feature_size)))
print("z shape",z.shape)

for line in fp:
    line = line.strip().split()
    z[int(line[0])][int(line[1])] = 1

for index in range(num_nodes):
    wp.write("{}".format(index))
    for value in z[index]:
        wp.write(" {}".format(value))
    wp.write("\n")

fp.close()
wp.close()

fp = open('edges.txt'.format(name))
wp = open("edges.txt1",'w')

edge_lst = []
for line in fp:
    line = line.strip().split()
    edge_lst.append(line)

for edge in edge_lst:
    wp.write("{} {}\n".format(edge[0],edge[1]))

fp.close()
wp.close()








