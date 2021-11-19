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

model = load_model('mymodel.h5')
#model.summary()
alph_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
             10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',
             20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',
             30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',36:'A',37:'B',38:'C',39:'D',
             40:'E',41:'F',42:'G',43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',
             50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',56:'U',57:'V',58:'W',59:'X',
             60:'Y',61:'Z'}
img= cv2.imread(inputfile)
img_copy =img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,400))
img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img_final = cv2.resize(img_thresh, (128,128))
img_final =np.reshape(img_final, (1,128,128,1))
img_pred = alph_dict[np.argmax(model.predict(img_final))]
print('Number of arguments: {}'.format(len(sys.argv)))
print('Argument(s) passed: {}'.format(str(sys.argv)))
cv2.putText(img, "Tipp: " + img_pred, (20,380),cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,0))
cv2.imshow('Handwritten character recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

