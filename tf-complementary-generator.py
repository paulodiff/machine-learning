# Complementary color generator
# https://www.mathsisfun.com/hexadecimal-decimal-colors.html

import csv
from random import randint

NUM_OF_ROW_DATA_TRAINING = 1000
NUM_OF_ROW_DATA_TEST = 100
NUM_OF_ROW_DATA_PREDICTION = 10

MAX_COLOR_VALUE = 16777215


# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(r, g, b):
    k = hilo(r, g, b)
    return list(k - u for u in (r, g, b))

def genColor():
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    # print(r,g,b)
    l = [r,g,b]
    return l

def color2num(c):
    # print((65536 * c[0] + 256 * c[1] + c[2])) 
    return (65536 * c[0] + 256 * c[1] + c[2]) / MAX_COLOR_VALUE

def num2col(n):
    nv = n * MAX_COLOR_VALUE
    # print('nv:', nv)
    r = int( nv / 65536 )
    # print(r * 65536)
    g = int ( (nv - ( r * 65536) )  / 256 )
    b = int ( (nv - ( r * 65536) - ( g  * 256 ) ) )
    return [r, g, b]

def generateData( numOfRow, fileName ):
    print('Generate', numOfRow, fileName )
    with open(fileName, 'wt') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(('SOURCE_COLOR','COMPLEMENT_COLOR'))
        for count in range(1, numOfRow):
            # print(count)
            t = genColor()    
            c = complement(t[0],t[1],t[2])
            # print(t[0],t[1],t[2],c[0],c[1],c[2])
    #       writer = csv.writer(csvfile, delimiter=',', quotechar='', quoting=csv.QUOTE_MINIMAL)
            # writer.writerow(   ('Title 1', 'Title 2', 'Title 3') )
            #writer.writerow((t[0],t[1],t[2],c[0],c[1],c[2]))
            writer.writerow( (color2num(t),color2num(c)) )
    csvfile.close()
    return

def generateDataPredict( numOfRow, fileName ):
    print('Generate', numOfRow, fileName )
    with open(fileName, 'wt') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(('SOURCE_COLOR','NONE'))
        for count in range(1, numOfRow):
            # print(count)
            t = genColor()    
            c = complement(t[0],t[1],t[2])
            # print(t[0],t[1],t[2],c[0],c[1],c[2])
    #       writer = csv.writer(csvfile, delimiter=',', quotechar='', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(  (color2num(t), color2num(c)) )
            #writer.writerow((t[0],t[1],t[2]))
    csvfile.close()
    return


test = genColor()    
cn = color2num(test)
nc = num2col(cn)
cp = complement(test[0],test[1],test[2])

print(test, cn, nc)    

#generateData(NUM_OF_ROW_DATA_TRAINING, 'tf-c-color-training.csv')
#generateData(NUM_OF_ROW_DATA_TEST, 'tf-c-color-test.csv')
generateDataPredict(NUM_OF_ROW_DATA_PREDICTION, 'tf-c-color-prediction.csv')

print('FINE OPERAZIONI')