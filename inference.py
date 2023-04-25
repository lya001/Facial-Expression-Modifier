import numpy as np
import cv2
from PIL import Image
import os
import glob
import tensorflow as tf
from StarGAN import StarGAN

def gen_image():
    for i in glob.glob('./input/*'):
        file_name = os.path.splitext(os.path.basename(i))[0]
        img = np.asarray(Image.open(i))

        # find face
        try:
            face = find_face(img)
        except:
            raise Exception("Couldn't find any face in the image.")

        # run model to manipulate facial expressions
        try:
            img_out = run_StarGAN(face)
        except:
            raise Exception("Error occurred while running the model.")

        img_out.insert(0, face)

        # save neutral image
        save_dir = './output/'
        neutral_index = 1
        pil_img = Image.fromarray(img_out[neutral_index].astype('uint8'))
        pil_img.save(save_dir + file_name + '.png')

def run_StarGAN(input_image):
    ''' config settings '''

    project_name = "StarGAN_Face_1_"
    train_flag = False

    '''-----------------'''

    # gpu_number = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" #args.gpu_number

    # with tf.device('/gpu:{0}'.format(gpu_number)):
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    #     config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.device('/cpu:0'):
        with tf.compat.v1.Session() as sess:
            model = StarGAN(sess, project_name)
            out_images = model.test(input_image, train_flag)
        tf.compat.v1.reset_default_graph()

    return out_images

def find_face(input_image):

    faceDet = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt_tree.xml")

    out_size = 128

    frame = input_image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    color_for_detection = gray

    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        raise Exception("No face detected in the image.")

    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        color = frame[y:y+h, x:x+w] #Cut the frame to size
        out = cv2.resize(color, (out_size, out_size)) #Resize face so all images have same size

    return out

if __name__ == "__main__":
    gen_image()
