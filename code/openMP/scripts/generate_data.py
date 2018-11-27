import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Read number of data points')
    parser.add_argument('-n', help='Number of data points')
    parser.add_argument('-m', help='Slope of best fit line')
    parser.add_argument('-b', help='Intercept of best fit line')
    args = parser.parse_args()
    return args

def generateNoise(numPoints):
    return np.random.normal(0, 10, 2*numPoints)

def generatePoints(numPoints):
    return np.random.uniform(0, 1000, numPoints)

def getY(x, m, b):
    return m * x + b

def generateData(numPoints, m, b):
    noise = generateNoise(numPoints)
    points = generatePoints(numPoints)
    x = []
    y = []
    for i in range(numPoints):
        x += [points[i] + noise[i]]
        y += [getY(points[i], m, b) + noise[i+1]]
    return x,y

def generateString(numPoints, m, b):
    x,y = generateData(numPoints, m, b)
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
    output = '%d.txt' % numPoints
    res = generateString(numPoints, slope, intercept)
    writeToFile(output, res)


if __name__ == '__main__':
    main()
