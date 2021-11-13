import cv2
import numpy as np
from keras.models import load_model
import sys,getopt


argv=sys.argv[1:]
inputfile=''
try:
    opts, args = getopt.getopt(argv,"hi:",["infile="])
except getopt.GetoptError:
    print('Alphabet_recogn.py -i <inputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Alphabet_recogn.py -i <inputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
print('Number of arguments: {}'.format(len(sys.argv)))
print('Argument(s) passed: {}'.format(str(sys.argv)))


model = load_model('model.h5')
#model.summary()
alph_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
img= cv2.imread(inputfile)
img_copy =img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,400))
img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))
img_pred = alph_dict[np.argmax(model.predict(img_final))]
cv2.putText(img, "Tipp: " + img_pred, (20,380),cv2.FONT_HERSHEY_SIMPLEX, 1.3, color = (255,0,255))
cv2.imshow('Handwritten character recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

