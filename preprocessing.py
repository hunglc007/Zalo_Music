# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from subprocess import Popen, PIPE, STDOUT
import os
import csv
from PIL import Image
import eyed3
from config import file_path,out_path,pixelPerSecond,desiredSize,slices_path,genresID,file_train_csv,slices_path_stride
def isMono(file_name):
    audiofile = eyed3.load(file_name)
    if audiofile.tag != None:
        print("meo nmeo")
    return audiofile.info.mode == 'Mono'
def readcsv(filepath):
    dataset = {}
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dataset[row[0]] = row[1]
    return dataset
def createData():
    currentPath = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
        except OSError as exc:
            raise
    train_csv = readcsv(file_train_csv)
    for root, subdirs, files in os.walk(file_path):
        for index, file in enumerate(files):
            split = file.split('.')
            if split[len(split) - 1] != 'mp3': continue
            file_name = file_path + "/" + file
            outFile_name = genresID[train_csv[file]] + "_" + str(split[0])
            if isMono(file_name):
                command = "cp '{}' '/tmp/{}.mp3'".format(file_name, outFile_name)
            else:
                command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(file_name, outFile_name)
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)
            file_name = file_name.replace(".mp3", "")
            command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(outFile_name,
                                                                                                       pixelPerSecond,
                                                                                               out_path + outFile_name)
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
                              cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)
            print(file_name)
def createSlices():
    filenames = os.listdir(out_path)
    filenames = sorted(filenames)
    if not os.path.exists(os.path.dirname(slices_path)):
        try:
            os.makedirs(os.path.dirname(slices_path))
        except OSError as exc:  # Guard again
            raise
    for filename in filenames:
        if filename.endswith(".png"):
            genre = filename.split("_")[0]
            img = Image.open(out_path + filename)

            slicePath = slices_path + "{}/".format(genre)
            if not os.path.exists(os.path.dirname(slicePath)):
                try:
                    os.makedirs(os.path.dirname(slicePath))
                except OSError as exc:  # Guard again
                    raise
            width, height = img.size
            nbSamples = int(width / desiredSize)
            for i in range(nbSamples):
                print( "Creating slice: ", (i + 1), "/", nbSamples, "for", filename)
                startPixel = i * desiredSize
                imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, 129))
                imgTmp.save(slicePath + "{}/{}_{}.png".format(genre, filename[:-4], i))
def createSlices_Stride():
    filenames = os.listdir(out_path)
    filenames = sorted(filenames)
    maxlen = len(os.listdir(slices_path + "TruTinh/"))
    listdir = [d for d in os.listdir(slices_path) if os.path.isdir(os.path.join(slices_path, d))]
    for idx, dir in enumerate(listdir):
        listfile = os.listdir(slices_path + '/' + dir)
        if len(listfile) < maxlen:
            countSample = len(listfile)
            start = 0
            while True:
                if countSample > maxlen: break
                for filename in filenames:
                    if filename.endswith(".png") and filename.split("_")[0] == dir:
                        genre = filename.split("_")[0]
                        img = Image.open(out_path + filename)

                        # Compute approximate number of 128x128 samples
                        width, height = img.size
                        nbSamples = int((width - start) / desiredSize)
                        countSample += nbSamples
                        if countSample > maxlen: break
                        # Create path if not existing
                        slicePath = slices_path_stride + "{}/".format(genre);
                        if not os.path.exists(os.path.dirname(slicePath)):
                            try:
                                os.makedirs(os.path.dirname(slicePath))
                            except OSError as exc:  # Guard again
                                raise
                        # For each sample
                        for i in range(nbSamples):
                            # print "Creating slice: ", (i + 1), "/", nbSamples, "for", filename
                            # Extract and save 128x128 sample
                            startPixel = i * desiredSize + start % 128
                            imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, 129))
                            print (slices_path_stride + "{}/{}_{}_slice{}.png".format(genre, filename[:-4], i, start / 10))
                            imgTmp.save(slices_path_stride + "{}/{}_{}_slice{}.png".format(genre, filename[:-4], i, start / 10))
            start += 10
if __name__ == '__main__':
    createData()
    createSlices()
    createSlices_Stride()
