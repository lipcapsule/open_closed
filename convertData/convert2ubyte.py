# to convert images to ubyte format on command prompt:
# py convert2ubyte.py test_data_folder test 
# py convert2ubyte.py train_data_folder train 

import numpy
import imageio
import glob
import sys
import os
import random

height = 0
width = 0

dstPath = "output"

testLabelPath = dstPath+"/t10k-labels-idx1-ubyte"
testImagePath = dstPath+"/t10k-images-idx3-ubyte"
trainLabelPath = dstPath+"/train-labels-idx1-ubyte"
trainImagePath = dstPath+"/train-images-idx3-ubyte"


def get_subdir(folder):
    listDir = None
    for root, dirs, files in os.walk(folder):
        if not dirs == []:
            listDir = dirs
            break
    listDir.sort()
    return listDir


def get_labels_and_files(folder, number=0):
    # Make a list of lists of files for each label
    filelists = []
    subdir = get_subdir(folder)
    for label in range(0, len(subdir)):
        filelist = []
        filelists.append(filelist)
        dirname = os.path.join(folder, subdir[label])
        for file in os.listdir(dirname):
            if (file.endswith('.jpg')):
                fullname = os.path.join(dirname, file)
                if (os.path.getsize(fullname) > 0):
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
        
        filelist.sort()  # sorts files to start off in same 

    # Build array of label/filename pairs
    labelsAndFiles = []
    for label in range(0, len(subdir)):
        count = number if number > 0 else len(filelists[label])
        filelist = random.sample(filelists[label], count)
        for filename in filelist:
            labelsAndFiles.append((label, filename))

    return labelsAndFiles


def make_arrays(labelsAndFiles, ratio):
    global height, width
    images = []
    labels = []
    imShape = imageio.imread(labelsAndFiles[0][1]).shape
    if len(imShape) > 2:
        height, width, channels = imShape
    else:
        height, width = imShape
        channels = 1
    for i in range(0, len(labelsAndFiles)):
        # display progress, since this can take a while
        if (i % 100 == 0):
            sys.stdout.write("\r%d%% complete" %
                             ((i * 100) / len(labelsAndFiles)))
            sys.stdout.flush()

        filename = labelsAndFiles[i][1]
        try:
            image = imageio.imread(filename)
            images.append(image)
            labels.append(labelsAndFiles[i][0])
        except:
            # If this happens we won't have the requested number
            print("\nCan't read image file " + filename)

    if ratio == 'train':
        ratio = 0
    elif ratio == 'test':
        ratio = 1
    else:
        ratio = float(ratio) / 100
    count = len(images)
    trainNum = int(count * (1 - ratio))
    testNum = count - trainNum
    if channels > 1:
        trainImagedata = numpy.zeros(
            (trainNum, height, width, channels), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width, channels), dtype=numpy.uint8)
    else:
        trainImagedata = numpy.zeros(
            (trainNum, height, width), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width), dtype=numpy.uint8)
    trainLabeldata = numpy.zeros(trainNum, dtype=numpy.uint8)
    testLabeldata = numpy.zeros(testNum, dtype=numpy.uint8)

    for i in range(trainNum):
        trainImagedata[i] = images[i]
        trainLabeldata[i] = labels[i]

    for i in range(0, testNum):
        testImagedata[i] = images[trainNum + i]
        testLabeldata[i] = labels[trainNum + i]
    print("\n")
    return trainImagedata, trainLabeldata, testImagedata, testLabeldata


def write_labeldata(labeldata, outputfile):
    header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
    global height, width
    header = numpy.array([0x0803, len(imagedata), height, width], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())


def main(argv):
    global idxLabelPath, idxImagePath
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    if len(argv) is 3:
        labelsAndFiles = get_labels_and_files(argv[1])
    elif len(argv) is 4:
        labelsAndFiles = get_labels_and_files(argv[1], int(argv[3]))
    random.shuffle(labelsAndFiles)

    trainImagedata, trainLabeldata, testImagedata, testLabeldata = make_arrays(
        labelsAndFiles, argv[2])

    if argv[2] == 'train':
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
    elif argv[2] == 'test':
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)
    else:
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)


if __name__ == '__main__':
    main(sys.argv)
