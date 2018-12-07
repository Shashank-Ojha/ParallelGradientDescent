import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Read number of data points')
    parser.add_argument('-n', help='Number of data points')
    parser.add_argument('-m', help='Slope of best fit line')
    parser.add_argument('-b', help='Intercept of best fit line')
    parser.add_argument('-std', help='Standard dev of best fit line')
    parser.add_argument('-r', help='Range of x values for data')
    args = parser.parse_args()
    return args

def generateNoise(numPoints, std):
    return np.random.normal(0, std, numPoints)

def generatePoints(numPoints, r):
    return np.random.uniform(0, r, numPoints)

def getY(x, m, b):
    return m * x + b

def generateData(numPoints, m, b, std, r):
    noise = generateNoise(numPoints, std)
    points = generatePoints(numPoints, r)
    x = []
    y = []
    for i in range(numPoints):
        x += [points[i]]
        y += [getY(points[i], m, b) + noise[i]]
    return x,y

def generateString(numPoints, m, b, std, r):
    x,y = generateData(numPoints, m, b, std, r)
    ret = '%d\n' % numPoints
    for i in range(numPoints):
        ret += '%f\t%f\n' % (x[i], y[i])
    return ret

def writeToFile(filename, ret):
    with open(filename, 'w') as file:
        file.write(ret)

def main():
    args = get_args()
    numPoints = int(args.n)
    slope = float(args.m)
    intercept = float(args.b)
    std = float(args.std)
    r = int(args.r)
    path = 'data/%.2f' % slope
    if not os.path.exists(path):
        os.makedirs(path)
    output = '%s/%d_%.2f_%d.txt' % (path, numPoints, std, r)
    res = generateString(numPoints, slope, intercept, std, r)
    writeToFile(output, res)


if __name__ == '__main__':
    main()
