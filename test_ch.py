import model
import reader
import numpy as np
import cv2 
import torch
from sklearn.metrics import confusion_matrix
import sys
import yaml
import os
import copy
import matplotlib.pyplot as plt

def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    config = config["test"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["load"]["model_name"] 

    loadpath = os.path.join(config["load"]["load_path"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder = os.listdir(labelpath)
    i = int(sys.argv[2])

    if i in range(2): 
        tests = folder[i]
        print(f"Test Set: {tests}")

        savepath = os.path.join(loadpath, f"checkpoint/{tests}")

        if not os.path.exists(os.path.join(loadpath, f"evaluation/{tests}")):
            os.makedirs(os.path.join(loadpath, f"evaluation/{tests}"))

        print("Read data")
        dataset = reader.txtload(os.path.join(labelpath, tests), imagepath, 10, shuffle=False, num_workers=4, header=True)

        begin = config["load"]["begin_step"]
        end = config["load"]["end_step"]
        step = config["load"]["steps"]

        # Initialize variables for confusion matrix
        all_predictions = []
        all_ground_truths = []

        for saveiter in range(begin, end+step, step):
            print("Model building")
            net = model.model()
            print(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))
            statedict = torch.load("E:/gazen/result/checkpoint/final/final.pt")

            net.to(device)
            net.load_state_dict(statedict)
            net.eval()

            print(f"Test {saveiter}")
            length = len(dataset)
            accs = 0
            count = 0
            with torch.no_grad():
                with open(os.path.join(loadpath, f"evaluation/{tests}/{saveiter}.log"), 'w') as outfile:
                    outfile.write("name results gts\n")
                    for j, (data, label) in enumerate(dataset):
                        print(f"Testing sample {j}...")
                        img = data["eye"].to(device) 
                        img_name = data["name"]
                        print("image", img_name)

                        img = {"eye": img}
                        gts = label.to(device)

                        gazes = net(img)

                        for k, gaze in enumerate(gazes):
                            gaze = gaze.cpu().detach().numpy()
                            if gaze[0] > (0.4)  and gaze[0] < (0.6):
                              print("CENTER")
                            elif gaze[0] > (0.7):
                               print("RIGHT")
                            elif gaze[0] < (0.4):
                                print("LEFT")
                            #all_predictions.append(gazeto3d(gaze))
                            #all_ground_truths.append(gazeto3d(gts.cpu().numpy()[k]))

                            count += 1
                            accs += angular(gazeto3d(gaze), gazeto3d(gts.cpu().numpy()[k]))

                            gaze = [str(u) for u in gaze] 
                            gt = [str(u) for u in gts.cpu().numpy()[k]]
                            print(f"Sample {j}, Subsample {k}: Predicted Gaze = [{gaze}], Ground Truth Gaze = [{gt}]") 
                            log = [",".join(gaze)] + [",".join(gt)]
                            outfile.write(" ".join(log) + "\n")

                    # Calculate confusion matrix
                    confusion_mat = confusion_matrix(all_ground_truths, all_predictions)

                    # Print or save confusion matrix
                    print("Confusion Matrix:")

                    print(confusion_mat)
                    # Plot confusion matrix as an image
                    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title('Confusion Matrix')
                    plt.colorbar()

                    # Add labels
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    # plt.xticks(np.arange(len(classes)), classes, rotation=45)
                    # plt.yticks(np.arange(len(classes)), classes)

                    # Display the plot
                    plt.show()

                    # You can also save the confusion matrix to a file if needed
                    np.savetxt(os.path.join(loadpath, f"evaluation/{tests}/{saveiter}_confusion_matrix.txt"), confusion_mat, fmt='%d')

                    # Calculate and print additional metrics if needed
                    # ...

                    loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
                    outfile.write(loger)
                    print("loger == ", loger)