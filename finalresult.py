import pickle
import csv
import numpy as np
from model import ResNet50
from PIL import Image
import os

def concatepklmodel(temp,pklmodel2,index):
    dict1 = temp
    with (open(pklmodel2, "rb")) as openfile:
        u = pickle._Unpickler(openfile)
        u.encoding = 'latin1'
        dict2 = u.load()
    count = 0
    for idx,value in dict2.items():
        if np.argmax(value) == index and value[index]/45 > 0.4:
            dict1[idx] = value
            count += 1
    print(count)
    return dict1
def concate_model(mode1,model2):
    dict1 = {}
    dict2 = {}
    with (open(mode1, "rb")) as openfile:
        dict1 = pickle.load(openfile)
    with (open(model2, "rb")) as openfile:
        u = pickle._Unpickler(openfile)
        u.encoding = 'latin1'
        dict2 = u.load()
    for idx,value in dict1.items():
        dict2[idx] = (dict2[idx] + value)/2
    newfinal = '/tmp/analyze_csv/concate06-2.42_06-4.90.pickle'
    with open(newfinal, 'wb') as handle:
        pickle.dump(dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)
def exportFinal():
    folder_final = '/result/'
    if not os.path.exists(os.path.dirname(folder_final)):
        try:
            os.makedirs(os.path.dirname(folder_final))
        except OSError as exc:  # Guard again
            raise
    fodler_analyze = '/tmp/analyze_csv/'

    modelroot = 'concate06-2.42_06-4.90.pickle'
    listmodel = {}
    listmodel[6] = ['29000_checkpoint.21-3.25.h5.pickle']
    listmodel[9] = ['10000_checkpoint.14-2.75.h5.pickle']
    with (open(fodler_analyze + modelroot, "rb")) as openfile:
        # dict1 = pickle.load(openfile)
        u = pickle._Unpickler(openfile)
        u.encoding = 'latin1'
        tempPickle = u.load()
    for idx,value in listmodel.items():
        for modelName in value:
            path = fodler_analyze + modelName
            tempPickle = concatepklmodel(tempPickle,path,idx)

    name = 'submission'
    newcsv = folder_final + name + '.csv'
    f = open(newcsv, 'w')
    writer = csv.writer(f)
    writer.writerow(["id", "genre"])
    for idx,value in tempPickle.items():
        f.write(idx + ".mp3," + str(np.argmax(value) + 1) + '\n')
    f.close()
def getProcessedData(img,imageSize):
    img = img.resize((imageSize,imageSize), resample=Image.ANTIALIAS)
    imgData = np.asarray(img, dtype=np.uint8).reshape(imageSize,imageSize,1)
    imgData = imgData/255.
    return imgData
def getImageData(filename,imageSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSize)
    return imgData
def final(modelPath, csvName):
    model = ResNet50()
    pickledir = '/tmp/analyze_csv/'
    if not os.path.exists(os.path.dirname(pickledir)):
        try:
            os.makedirs(os.path.dirname(pickledir))
        except OSError as exc:  # Guard again
            raise
    modelPath = modelPath
    slicesPathTest = '/tmp/SlicesPrivateTest'
    model.load_weights(modelPath)
    finalanalyze = pickledir + csvName
    analyze_dict = {}
    filenames = os.listdir(slicesPathTest)
    sliceSize = 128
    batchsize = 16
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    temp = ''
    x_batch = []
    y_batch = []
    count_batch = 0
    tempresult = np.zeros([10])
    filenames = sorted(filenames)
    for idx,filename in enumerate(reversed(filenames)):
        images = []
        imgData = getImageData(slicesPathTest + "/" + filename, sliceSize)
        images.append(imgData)
        images = np.asarray(images)
        if filename.split('_')[0] == temp:
            if count_batch < batchsize:
                count_batch += 1
                x_batch.append(imgData)
            else:
                x_batch = np.array(x_batch)
                result = model.predict(x_batch)
                result = np.sum(result, axis=0)
                result = np.reshape(result, (10))
                tempresult += result
                count_batch = 0
                x_batch = []
        else:
            if(temp != ''):
                if count_batch != 0:
                    x_batch = np.array(x_batch)
                    result = model.predict(x_batch)
                    result = np.sum(result, axis=0)
                    result = np.reshape(result, (10))
                    tempresult += result
                    count_batch = 0
                    x_batch = []
                idx = np.argmax(tempresult) + 1
                analyze_dict[temp] = tempresult
                print (temp)
            temp = filename.split('_')[0]
            tempresult = np.zeros([10])
            x_batch.append(imgData)
            count_batch += 1
    x_batch = np.array(x_batch)
    result = model.predict(x_batch)
    result = np.sum(result, axis=0)
    result = np.reshape(result, (10))
    tempresult += result
    idx = np.argmax(tempresult) + 1
    analyze_dict[temp] = tempresult
    with open(finalanalyze + '.pickle', 'wb') as handle:
        pickle.dump(analyze_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print (modelPath)

if __name__ == '__main__':
    rootdirectory = '/media/vnc/Other/ZaloChallenge/Music/models'
    listfile = os.listdir(rootdirectory)
    for file in listfile:
        final(rootdirectory + '/' + file, file)
    model1path = '/tmp/analyze_csv/fulldata_checkpoint.06-4.90.h5.pickle'
    model2path = '/tmp/analyze_csv/29000_checkpoint.06-2.42.h5.pickle'
    concate_model(model1path,model2path)
    exportFinal()


