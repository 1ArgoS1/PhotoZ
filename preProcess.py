import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

file_path = "../data/cube.npy"
label_path = "../data/labels.npy"

data = np.load(file_path,mmap_mode='r')
labels = np.load(label_path, mmap_mode='r')
chunk_count = 20 #change here to alter file size.
chunk_s = data.shape[0]//chunk_count

#temp = 0
for i in tqdm(range(3)):
    if(i == chunk_count):
        #print(data[(i*chunk_s):data.shape[0] ].shape)
        #temp += data[(i*chunk_s):data.shape[0] ].shape[0]
        np.save("cube_part_"+str(i),data[(i*chunk_s):data.shape[0]])
        np.save("labels_part_"+str(i),labels[(i*chunk_s):data.shape[0]])


    else:
        #print(i*chunk_s,(i+1)*chunk_s - 1 )
        #print(data[(i*chunk_s):((i+1)*chunk_s)].shape)
        #temp += data[(i*chunk_s):((i+1)*chunk_s-1)].shape[0]
        np.save("cube_part_"+str(i),data[(i*chunk_s):((i+1)*chunk_s)])
        np.save("labels_part_"+str(i),labels[(i*chunk_s):((i+1)*chunk_s)])
#print(temp , data.shape[0])

