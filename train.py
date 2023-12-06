# import model
import model
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    config = config["train"]
    # print("config===",config)
    cudnn.benchmark=True
  
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]
    print("imagepath===",imagepath)
    print("labelpath===",labelpath)
    folder = os.listdir(labelpath)
    print("folders===",folder)
    folder.sort()

    # i represents the i-th folder used as the test set.
    i = int(sys.argv[2])
    print("sysss===",i)

    # if i in list(range(5)):
    trains = copy.deepcopy(folder)
    tests = copy.deepcopy(folder)
    print(f"Train Set:{trains}")
    print(f"Test Set:{tests}")

    trainlabelpath = [os.path.join(labelpath, j) for j in trains] 
    print("trainlabel===",trainlabelpath)

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint/{tests}")
    if not os.path.exists(savepath):
      os.makedirs(savepath)
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Read data")
    dataset = reader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=False, num_workers=0, header=True)
    print("HUAAAAAAA")
    if  dataset is  None:
       print("DATASET NOT AVIALABLE")

    print("Model building")
    net = model.model()
    net.train()
    net.to(device)

    print("optimizer building")
    lossfunc = config["params"]["loss"]
    loss_op = getattr(nn, lossfunc)().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Traning")
    length = len(dataset)
    print("dataset lenght == ",length)
    total = length * config["params"]["epoch"]
    print("total == ",total)
    cur = 0
    timebegin = time.time()
    train_losses = []
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
      for epoch in range(1, config["params"]["epoch"]+1):
        total_loss =0.0
        for index, (data, label) in enumerate(dataset):
          data["eye"] = data["eye"].to(device)  # image
          label = label.to(device)
          print("HEAD2D==",data['head_pose']) 
          print("LABEL",label)
          gaze = net(data)
          gaze = gaze.to(device)  # Move gaze to the same device as data["eye"]
          print("PREDICTION ===",gaze)
          loss = loss_op(gaze, label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          scheduler.step()
          cur += 1
          total_loss += loss.item()
       
          # print logs
          if i % 20 == 0:
            timeend = time.time()
            resttime = (timeend - timebegin)/cur * (total-cur)/3600
            log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
            print(log)
            outfile.write(log + "\n")
            sys.stdout.flush()   
            outfile.flush()
        average_loss = total_loss / (3000)
        train_losses.append(average_loss)

        if epoch  == 5:
          print("I am saving !!!")
          torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))
          # Plot the epoch vs loss graph
    plt.plot(range(1, config["params"]["epoch"]+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.show()