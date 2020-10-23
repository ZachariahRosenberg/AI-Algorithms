import os, sys, cv2
import matplotlib.pyplot as plt

def detect_face(img_path):
    '''
    Uses Haar Cascades to detect and draw a bounding box around faces. read more here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
    '''

    # OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # Convert image to greyscale
    img  = cv2.imread(img_path)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Locate faces in image
    faces = face_cascade.detectMultiScale(grey)

    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)

    # Bring color back to image
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display the image, along with bounding box
    plt.imshow(img_color)
    plt.show()


if __name__ == "__main__":

    img_path = sys.argv[1]

    # Check if img_path file exists
    valid_photo_extensions = ['.png', 'jpeg', '.jpg', '.gif', '.bmp'] #quick hack way to check if file is photo
    if not os.path.isfile(img_path) or not img_path[-4:] in valid_photo_extensions:
        raise Exception("Image path invalid. Make sure to include a path to a valid image file (png, jpg, gif)")

    detect_face(img_path)
