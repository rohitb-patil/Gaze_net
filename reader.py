import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch

split_values = []
def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  print("yaw==",yaw)
  print("pitach",pitch)
  return np.array([yaw, pitch])

class loader(Dataset): 
  m =0
  
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root


  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    if idx >= len(self):
        raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")
    line = self.lines[idx]
    line = line.strip().split(" ")
    # print("line===",line)
    name = line[1]
    # print("name===",name)
    gaze2d = line[5]   # 2 d normalised points
    # print("gaze2d===",gaze2d)
    head2d = line[6] # head rotation
    # print("head2d===",head2d)
    eye = line[0]    #name of a image
    onscreen = line[10]
    # print("shape of eye===",eye.shape())
    # print("gaze2d=",gaze2d)
    # print("head rotation ==",head2d)
    # print("image == ",eye)
    # print("\n")
    try:
     label = np.array(gaze2d.split(",")).astype("float")
     label = torch.from_numpy(label).type(torch.FloatTensor)
     print("LABELL==",label)
    except ValueError:
     print(f"Error converting '{gaze2d}' to float.")
     label = np.array([0.0, 0.0])  # Example default value

     ###########################
    screen = np.array(onscreen.split(",")).astype("float")
    screen[0] /= 1280.0
    screen[1] /= 800.0
    screen = torch.from_numpy(screen).type(torch.FloatTensor)
     
    # print("label==",type(label))
    
    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)
    # print("headpose==",type(headpose))
    # print("===============================================================================")
    img = cv2.imread(os.path.join(self.root, eye))/255.0
    print("input image shape ROOOOOT ==",self.root)
    img = img.transpose(2, 0, 1)
    print("SHAPE OF IMAGE ===========",img.shape)

    info = {"eye":torch.from_numpy(img).type(torch.FloatTensor),
            "head_pose":screen,  # chnages made(headpose ->label) 
            "name":name,
            }
    print("shape of eye ",img.shape) # the resolution is fine 224x224
    return info, screen

def txtload(labelpath, imagepath, batch_size, shuffle=False, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  # print("DATASET====",dataset)
  # print(f"[Read Data]: Total num: {len(dataset)}")
  # print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print("lenght===",len(d))
  (data, label) = d.__getitem__(0)
  print("type",type(data),type(label))