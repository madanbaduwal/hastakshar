import os
from tqdm import tqdm
import cv2
import csv
import ast
import numpy as np
# import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import backend as K
import hashlib
import tensorflow as tf
import time

# import pycuda.autoinit
# from pycuda import gpuarray
# import libcudnn, ctypes



from keras import layers
from keras import models
from keras import optimizers

EPOCHS=20
train_path = './verification/trainingset'
x_genuine_path = './verification/genuine'
x_forgery_path = './verification/forgery'
x_random_path = './verification/random'
x_random_csv_path = './verification/randomcsv'  
x_test_path = './verification/test'
save_path = './verification/modelsave'

graph = tf.get_default_graph()

def segmentImage(image):  
  hHist=np.zeros(shape=image.shape[0], dtype=int)
  vHist=np.zeros(shape=image.shape[1], dtype=int)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if(image[i][j]==0):
        hHist[i]+=1
        vHist[j]+=1
  
  locLeft=0
  locRight=image.shape[0]
  locTop=0
  locBottom=image.shape[1]
  
  count=0
  for i in range(hHist.shape[0]):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locTop=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(hHist.shape[0])):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locBottom=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in range(vHist.shape[0]):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locLeft=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(vHist.shape[0])):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locRight=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  return locLeft, locRight, locTop, locBottom
			

def preProcessImage(train_path, final_img_size = (300,300), power_law=False, segment=True, log_transform=False):
  train_batch = os.listdir(train_path)
  x_train = []
  train_data = train_batch
  #train_data = [x for x in train_batch if x.endswith('png') or x.endswith('PNG') or x.endswith('jpg') or x.endswith('JPG') or x.endswith('TIF') or x.endswith('tif')]

  for sample in tqdm(train_data):
    img_path = os.path.join(train_path, sample)
    #importing images from drive
    #x = image.load_img(img_path)
    #img = image.img_to_array(x)
    img = cv2.imread(img_path)
        
    #Perfom Median blur on image
    mbvalue = int(np.max(img.shape)/200)
    mbvalue = mbvalue if mbvalue%2==1 else mbvalue+1
    img = cv2.medianBlur(img, mbvalue)

    #changing RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         
    #resize image to 600xH
    img = cv2.resize(img, (600, int(600*float(img.shape[0])/img.shape[1])))
    
    #if power_law enabled
    if(power_law):
      img = img**0.9
      img[img>255]=255
      img[img<0]=0
      img = img.astype('uint8')
          
    #denoising the grayscale image
    img = cv2.fastNlMeansDenoising(img, None, 10, 21)
    
    if (log_transform):
        img = (np.log(img+1)/(np.log(10+np.max(img))))*255
        img=img.astype('uint8')
    
    #Threshold binary image
    # avg = np.average(img)
    # _,image = cv2.threshold(img, int(avg)-30, 255, cv2.THRESH_BINARY)
    
    #New Steps in thresholding
    dst = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
  
    img = dst-img
    #Threshold binary image
    _,image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    image = cv2.medianBlur(image, 3)

    #segment the signature section only
    if(segment):
      seg = segmentImage(image)
      image = image[seg[2]:seg[3], seg[0]:seg[1]]
          
    #padding to make image into square
    lp, rp, tp, bp = (0,0,0,0)
    if(image.shape[0]>image.shape[1]):
      lp = int((image.shape[0]-image.shape[1])/2)
      rp = lp
    elif(image.shape[1]>image.shape[0]):
      tp = int((image.shape[1]-image.shape[0])/2)
      bp = tp
    image_padded = cv2.copyMakeBorder(image, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=255)

    #resizing the image
    img = cv2.resize(image_padded, final_img_size)

    #producing image negative
    img = 255-img

    #skeletonizing image
    #img = thin(img/255)

    img = img.astype('bool')

    #appending it in list
    x_train.append(img)

  #converting it into np-array  
  x_train = np.array(x_train)
  return x_train

def convertToInt(arr):
  t1=[]
  for x in arr:
    t2=[]
    for y in x:
      t3=[]
      for z in y:
        if(z==True):
          t3.append(1)
        else:
          t3.append(0)
      t2.append(np.array(t3))
    t1.append(np.array(t2))
  return np.array(t1).astype('uint8')

def convertToBool(arr):
  t1=[]
  for x in arr:
    t2=[]
    for y in x:
      t3=[]
      for z in y:
        if(z==1):
          t3.append(True)
        else:
          t3.append(False)
      t2.append(np.array(t3))
    t1.append(np.array(t2))
  return np.array(t1)


def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)


def csvReader(fil_name):
  with open(fil_name+'.csv', 'r') as f:
    reader = csv.reader(f)
    examples = list(reader)
    examples = np.array(examples)
  
  t1=[]
  for x in examples:
    t2=[]
    for y in x:
      z= ast.literal_eval(y)
      t2.append(np.array(z))
    t1.append(np.array(t2))
  ex = np.array(t1)
  return ex

def model():
    # old model
    # mod = models.Sequential()
    # mod.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(300, 300, 1)))
    # mod.add(layers.MaxPooling2D((2, 2)))
    # mod.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # mod.add(layers.MaxPooling2D((2, 2)))
    # mod.add(layers.Flatten())
    # mod.add(layers.Dropout(0.5))
    # mod.add(layers.Dense(256, activation='relu'))
    # mod.add(layers.Dropout(0.5))
    # mod.add(layers.Dense(128, activation='relu'))
    # mod.add(layers.Dense(1, activation='sigmoid'))
    #
    # mod.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])

    # New model
    mod = models.Sequential()
    mod.add(layers.Conv2D(16, (9, 9), activation='relu', input_shape=(300, 300, 1)))
    mod.add(layers.MaxPooling2D((2, 2)))
    mod.add(layers.Conv2D(32, (5, 5), activation='relu'))
    mod.add(layers.MaxPooling2D((2, 2)))
    mod.add(layers.Conv2D(32, (3, 3), activation='relu'))
    mod.add(layers.MaxPooling2D((3, 3)))
    mod.add(layers.Conv2D(16, (2, 2), activation='relu'))
    mod.add(layers.MaxPooling2D((2, 2)))
    mod.add(layers.Flatten())
    mod.add(layers.Dropout(0.2))
    mod.add(layers.Dense(256, activation='relu'))
    mod.add(layers.Dropout(0.4))
    mod.add(layers.Dense(128, activation='relu'))
    mod.add(layers.Dense(1, activation='sigmoid'))

    mod.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])

    return mod
    print('********Done!!****')

def getmd5(filename):
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()

    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    print("##############################")
    print("MD5: {0}".format(md5.hexdigest()))
    print("##############################")
    return md5.hexdigest()

def train(filename):
    mod= model()
    # if 'mysign_weights.h5' not in os.listdir(save_path):

    x_genuine = preProcessImage(x_genuine_path)
    x_random = csvReader(os.path.join(x_random_csv_path, 'random3'))
    # another training
    mod.fit(np.concatenate((x_genuine, x_random)).reshape(x_genuine.shape[0] + x_random.shape[0], 300, 300, 1),
            np.concatenate((np.full(x_genuine.shape[0], 1), np.full(x_random.shape[0], 0))), epochs=EPOCHS, verbose=1,
            shuffle=True)

    evaluated = mod.evaluate(x_genuine.reshape(x_genuine.shape[0], 300, 300, 1), np.full((x_genuine.shape[0]), 1))
    print('Accuracy: ', evaluated[1] * 100, '%')

    mod.save_weights(os.path.join(save_path, filename+'.h5'))

    time.sleep(1)

    return getmd5(os.path.join(save_path, filename+'.h5'))
    # mod.save_weights(filename+'.h5')

    # else:
    #     mod.load_weights(os.path.join(save_path, 'mysign_weights.h5'))


def test(filename):
    K.clear_session()
    global graph
    mods= model()
    mods.load_weights(os.path.join(save_path, filename+'.h5'))
    mods._make_predict_function()
    print("$$$$$$$$$$$$$ HAS LOADED $$$$$$$$$$")
    x_test = preProcessImage(x_test_path)
    x_test = x_test.reshape(300, 300, 1)
    with graph.as_default():
        # predicted_acc = mod.predict(np.array([x_test.reshape(300, 300, 1), ]))
        predicted_acc = mods.predict(np.array([x_test, ]))
    print("\n\nAccuracy of Sign in Test Folder: ", predicted_acc * 100, "%")
    print("*** If percent > 90% accept the signature ***")
    K.clear_session()
    return predicted_acc[0][0]*100

'''
history = mod.history.history
plt.plot(history['acc'], marker='o', linewidth=3, color='blue', label='Accuracy')
plt.plot(history['loss'], marker='X', linewidth=3, color='red', label='Loss')
plt.plot(history['val_acc'], marker='o', linewidth=3, color='green', label='Val_Acc')
plt.plot(history['val_loss'], marker='X', linewidth=3, color='brown', label='Val_Loss',)
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()
plt.savefig('history3.png', bbox_inches='tight', dpi=200)
'''
