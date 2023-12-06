import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
sys.path.append("../core/")
import data_processing_core as dpc

root = "E:/work_gaze/MPIIGaze/MPIIGaze"
sample_root = "E:/work_gaze/MPIIGaze/MPIIGaze/Evaluation Subset/sample list for eye image"
out_root = "E:/work_gaze/f_data"
scale =True

def ImageProcessing_MPII():
    persons = os.listdir(sample_root)
    persons.sort()
    for person in persons:
        sample_list = os.path.join(sample_root, person) 
        print("sample =",sample_list)

        person = person.split(".")[0]
        print("person==",person)
        im_root = os.path.join(root,  "Data", "original", person)
        print("root==",im_root)

        im_outpath = os.path.join(out_root, "Image", person)
        print("SAVING PATH ",im_outpath)
        label_outpath = os.path.join(out_root, "Label", f"{person}.label")
        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))

        print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, sample_list, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, sample_list, im_outpath, label_outpath, person):
    days = os.listdir(im_root)
   
    # Read camera matrix
    days.remove("Calibration")
    camera = sio.loadmat(os.path.join(f"{im_root}", "Calibration", "Camera.mat"))
    camera = camera["cameraMatrix"]

    # Read gaze annotation
    anno_dict = {}
    for day in days:
        path = os.path.join(im_root, day)
        contents = os.listdir(path)
        im_num = len(contents) - 1
        
        annotation = os.path.join(path, "annotation.txt")
        # print("what is in anno ==",annotation)
        with open(annotation) as infile:
            anno_mes = infile.readlines()
        # print("DAY==",day)
        anno_dict[day] = anno_mes
        # print("DIC ==",anno_dict["day01"])
        # assert len(anno_mes) == im_num, print("The length of annotatioin is not equal to number of image.")

    # Create the handle of label 
    outfile = open(label_outpath, 'w')
    outfile.write("Image Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin Onscreen\n")

    # Image Processing 
    with open(sample_list) as infile:
        im_list = infile.readlines()
        total = len(im_list)
    # print("TOTAL == ",total)

    for count, info in enumerate(im_list):
        # print("count =",count,"info",info)

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/total * 20))
        progressbar = "\r" + progressbar + f" {count}|{total}"
        print(progressbar, end = "", flush=True)

        # Read image info
        im_info, which_eye = info.strip().split(" ") # day08/0069.jpg    left 
        print("IMAGE KONSA ==",im_info)
        day, im_name = im_info.split("/")  #day08    0069.jpg
        im_number = int(im_name.split(".")[0])   #0069 
        # day08/0069.jpg left
        # Read image annotation and image
        im_path = os.path.join(im_root, day, im_name)
        # print("IM_PATH ===============================",im_path)
        im = cv2.imread(im_path, 0)
        print("SIZE first ====",im.shape)
        annotation = anno_dict[day][im_number-1]
        print("ANOODICT ==",annotation)
        # print("DICT ==",day,im_number)
        # print("ANNOOO ==",annotation)
        print("\n")
        annotation = AnnoDecode(annotation)
        # print("decode ==",annotation)

        # Normalize the image
        if which_eye == "left":
            norm = dpc.norm(center = annotation["leftcenter"],
                            gazetarget = annotation["target"],
                            headrotvec = annotation["headrotvectors"],
                            imsize = (60, 36),
                            camparams = camera)
            print("GAZETARGET left ==",annotation["leftcenter"])
            origin = norm.GetCoordinate(annotation["leftcenter"])
            print("LEFT ==",origin)
     
        else:
            norm = dpc.norm(center = annotation["rightcenter"],
                            gazetarget = annotation["target"],
                            headrotvec = annotation["headrotvectors"],
                            imsize = (60, 36),
                            camparams = camera)
            print("GAZETARGET right ==",annotation["rightcenter"])
            origin = norm.GetCoordinate(annotation["rightcenter"])
            # print("RIGHT ==",norm)

        # Acquire essential info
        im_eye = norm.GetImage(im)
        print("IM EYE ======",im_eye.shape)
        im_eye = cv2.equalizeHist(im_eye)
        print("HISTooo",im_eye.shape)
        # print("HIST  ======",im_eye.shape)
        gaze = norm.GetGaze(scale=scale)
        head = norm.GetHeadRot(vector=True)

        if which_eye == "left":
            pass
        else:
            im_eye = cv2.flip(im_eye, 1)
            gaze = dpc.GazeFlip(gaze)
            head = dpc.HeadFlip(head) 
            origin[0] = -origin[0]
        gaze_2d = dpc.GazeTo2d(gaze)
        head_2d = dpc.HeadTo2d(head)
        
        rvec, svec = norm.GetParams()

        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, str(count+1)+".jpg"), im_eye)
        print("AFTER ===",im_eye.shape)
        
        save_name = os.path.join(person, str(count+1) + ".jpg")
        save_origin = im_info
        save_flag = which_eye
        print("\n")
        print("konsa eye ==",which_eye)
        save_gaze = ",".join(gaze.astype("str"))
        save_head = ",".join(head.astype("str"))
        save_gaze2d = ",".join(gaze_2d.astype("str"))
        save_head2d = ",".join(head_2d.astype("str"))
        screen = annotation["onscreen"]
        save_screen = ",".join(screen.astype("str"))
        
        save_rvec = ",".join(rvec.astype('str')) 
        save_svec = ",".join(svec.astype('str')) 
        origin = ",".join(origin.astype('str')) 
        
        save_str = " ".join([save_name, save_origin, save_flag, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, origin,save_screen])
        
        outfile.write(save_str + "\n")
    print("")
    outfile.close()


def AnnoDecode(anno_info):
    annotation = np.array(anno_info.strip().split(" ")).astype("float32")
    out = {}
    out["left_left_corner"] = annotation[0:2]
    out["left_right_corner"] = annotation[6:8]
    out["right_left_corner"] = annotation[12:14]
    out["right_right_corner"] = annotation[18:20]
    out["headrotvectors"] = annotation[29:32]
    out["headtransvectors"] = annotation[32:35]
    out["rightcenter"] = annotation[35:38]
    out["leftcenter"] = annotation[38:41]
    out["target"] = annotation[26:29]
    
    # Add a new key "onscreen" and assign the desired range in annotation
    out["onscreen"] = annotation[24:26]  # Adjust indices as needed

    return out


if __name__ == "__main__":
    ImageProcessing_MPII()
