"""works only for a specific input format. Highly unpredictable at the moment. """

"""This is a Python script to parse the log file generated by CARLSim."""


import re
import struct
import math
import sys, os, getopt
# Parsing Number of input and output neurons
#input_layer = 200
#hidden_layer = 

#create a list of number of groups and the number of neurons per group. 
layer = []

groupName = ['input', 'smoothExc', 'smoothInh', 'edges']
connections = [[0,1], [1, 2]]

iterator = [0, 784, 196]

# indicates the spike monitor groups - configuration in CARLSIM.
groupNumber = [0, 1, 2]

#number of connections to be monitored 
connectionNumber = 2 


def neuronNumberParser(log_str):
    """Return (input_number, output_number)."""
    
    input_number_re = re.compile('(\Num\sof\sNeurons\s+=\s+[0-9]+)+', flags=0)
    input_number = input_number_re.findall(log_str, 2)
    for pos,i in enumerate(input_number):
        neuNumber = re.search('[0-9]+', i).group()
        layer.append(int(neuNumber))


# Parsing the connection table
# The return value is a dictionary {int, [int(s)]}.
# The first integer is input neuron ID
# and the array contains all the node connected.

def connectionTableParser(log_str):
    """Return the array of connection pairs."""
    # search for the start of connection monitor information first
    connectionTable = {} 
    weightTable = {}
    
    pattern_1 = re.compile('ConnectionMonitor', flags=0)
    pattern = re.compile('(\[\s*[0-9]+,\s*[0-9]+\]\s(([0-9]|\.)+|nan)\s+\n?)+',flags=0)
    sourceOffset = 0
    destOffset = 0
    itr = 0
    while itr in range(0, len(log_str)):
        line = log_str[itr]
        if pattern_1.search(line): 
            #print(line)
            patternFound = 0
            for i in connections:
                connPattern = re.compile('ConnectionMonitor\s\ID=[0-9]\s{}\([A-za-z]+\)\s\=>\s{}'.format(i[0], i[1]), flags=0)#{}([A-Z]+[a-z]+)\s\=>\s{}'.format(i[0], i[1]), flags=0)
                if connPattern.search(line):
                    #print(line)
                    sourceOffset = i[0]
                    destOffset = i[1]
                    itr = itr+1
                    line = log_str[itr]
                    patternFound = 1 
                    break
            
            if not patternFound: 
                itr = itr + 1
                line = log_str[itr]

            while not pattern_1.search(line) and patternFound:
                SRE_MATCH_TYPE = type(re.match("", ""))
                if type(pattern.search(line)) is SRE_MATCH_TYPE:
                    connectionTableList = pattern.search(line).group()
                    weight = re.compile('(\[\s*[0-9]+,\s*[0-9]+\]+)', flags=0).sub('',connectionTableList).replace("\n", " ")
                    weight = re.sub(' +', ' ', weight).strip().split(' ')
                    connectionTableList = re.compile('\s(nan|([0-9]+\.[0-9]+))\s+', flags=0).sub(';', connectionTableList).strip()
                    connectionTableList = re.compile('\s+', flags=0).sub('', connectionTableList).split(';')
                    connectionTableList.pop() 
                    for pos,pair in enumerate(connectionTableList): 
                        pairPattern = re.compile('[0-9]+,[0-9]+').search(pair).group()
                        pairPattern = re.split(',', pairPattern)
                        key = int(pairPattern[0])
                        value = int(pairPattern[1])
                        weightKey = (key,value)
                        if not math.isnan(float(weight[pos])):
                            #print(key,value)
                            # store the weights of a key
                            weightTable[weightKey] = float(weight[pos])  
                            # store the srcNeuron(key) -> destNeuron(value) in a list. 
                            if key in connectionTable:
                                connectionTable[key].append(value)
                            else:
                                connectionTable[key] = [value] 
                        else:
                            weightTable[weightKey] = float(0)


                    for key in connectionTable: 
                        sourceNeuron = key + iterator[i[0]] * i[0]
                        connection_table_file.write('%d\n' % sourceNeuron)
                        #print(sourceNeuron)
                        for value in connectionTable[key]: 
                            destinationNeuron = value + iterator[i[1]] # * i[1]
                            #print(destinationNeuron)
                            connection_table_file.write('%d ' % destinationNeuron)
                        connection_table_file.write('\n\n')

                    for key in weightTable: 
                        sourceNeuron = key[0] + iterator[sourceOffset]
                        destNeuron = key[1] + iterator[destOffset]
                        weight = weightTable[key]
                        weight_table_file.write('%d %d %f' % (sourceNeuron, destNeuron, weight))
                        weight_table_file.write('\n')
            
                    connectionTable.clear()
                    weightTable.clear()
                    itr = itr + 1
                    line = log_str[itr]
                    #print(line)
                else:
                    break
        else:
            itr = itr + 1
            #print(itr)
            if itr in log_str:
                line = log_str[itr]
    #print(weightTable)                
# Parsing the spiking times

def spikingTimeParser(log_str, filePath):
    #"""Return the spiking times of each neuron."""
    #find all the seperate groups first/ differentiate between the groups first
    
    first_line_re = '\|\s*[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*'
    rest_line_re = '\|\s+\|\s+\|(\s*[0-9]+)+\s+'
    spikeMap = {}
    position = 0
    itr = 0   
    

    while itr in range(0, len(log_str)):
        grpNum = 0
        groupNumberVal = 0
        line = log_str[itr]
        spikeMonitor_re = re.compile('SpikeMonitor for group', flags=0)
        position = 0
        breakLine_re = 'ConnectionMonitor ID='
        breakLine = re.compile(breakLine_re, flags=0)
        if breakLine.search(line):
            break

        if spikeMonitor_re.search(line):
            for grpNum in groupNumber: 
                groupNumber_re = re.compile('SpikeMonitor for group [A-za-z]+\({}\)'.format(grpNum), flags=0)#[A-za-z]+({})
                if groupNumber_re.search(line):
                    groupNumberVal = int(grpNum)
                    position = position + iterator[groupNumberVal] 
                    #print(position)

            pattern1 = re.compile(first_line_re, flags=0)
            itr = itr+1
            line = log_str[itr]
            while not pattern1.search(line):
                itr = itr+1
                line = log_str[itr]                    
            
            #print("reached here")
            while pattern1.search(line) :
                timeTable = pattern1.search(line)
                timeList = timeTable.group().replace('|', '').strip()
                timeList = re.compile('\s+', flags=0).split(timeList)
                neuronId = int(timeList.pop(0)) + position
                spikeRate = float(timeList.pop(0))
                spikeMap[neuronId] = []
                #print(neuronId)
                if spikeRate > 0:
                    second_line_re = '\|\s*[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|([\s]*[0-9]+)+\s+'
                    secondLine = re.compile(second_line_re, flags = 0)
                    line = log_str[itr]
                    spikeTimes = secondLine.search(line).group().replace('|', '').strip()
                    spikeTimes = re.compile('\s+', flags = 0).split(spikeTimes)
                    spikeTimes.pop(0) 
                    spikeTimes.pop(0)
                    for spike in spikeTimes:
                        spike = int(spike)
                        spikeMap[neuronId].append(spike)

                itr = itr+1
                line = log_str[itr]
                pattern2 = re.compile(rest_line_re, flags=0)
                while pattern2.search(line):
                    timeTable = pattern2.search(line)
                    timeList = timeTable.group().replace('|', ''). strip()
                    timeList = re.compile('\s+', flags = 0).split(timeList)
                    for spike in timeList:
                        spike = int(spike)
                        spikeMap[neuronId].append(spike)
                    itr = itr+1
                    line = log_str[itr]

                #print(spikeMap[neuronId])                   
        else:
            itr = itr + 1
                
    #print(spikeMap)

    #trafficFile = open('../traffic.txt', mode='w')
    spikeCount = open(filePath+'spike_info.txt', mode='w')

    for key in spikeMap:
        #outputFile.write('\n\nNeuron %d spikes at\n' % key)
        spikeCount.write('%d,%d\n' % (key,len(spikeMap[key])))
        for i in spikeMap[key]:
            spikeCount.write('%d ' % (i))
            #outputFile.write('%d, ' % i)
        spikeCount.write('\n')

    spikeCount.close()
    

def testFormat(argv):

    inputfile = ''
    outputfile = ''

    if len(argv) > 1:
        if argv[1] == '-h':
            print 'log_parser.py <inputfile> <outputFilePath>'
            sys.exit()

    if(len(argv) != 3):
        print 'Input format is ---> python<v> log_parser.py <inputfile> <outputfilePath>'
        sys.exit(2)

    else:
        inputfile = argv[1]
        outputfile = argv[2]

    print 'Input file is "', inputfile
    print 'Output file is "', outputfile
    return inputfile, outputfile






if __name__ == "__main__":
    inputFile, outputFilePath = testFormat(sys.argv)


    # Build traffic table file
    # Format:
    # [Spiking time]_[Group ID]_[Neuron ID]

    log_file1 = open(inputFile, mode='r')  #'/home/adarsha/Research/Noxim/noxim-extended-for-neuromorphic/bin/carl-logs/8x8.log', mode='r')
    log_file_lines1 = log_file1.readlines()
    spiking_time = spikingTimeParser(log_file_lines1, outputFilePath)

    log_file1.close()


    # Build connection table file
    # Format:
    # [Number of groups (currently is set to 2)]
    # Iteration{
    # [Source group ID]_[Source neuron ID]_[Number of destinations]
    # [Destination group ID (currently is set to 1)]
    # [Destination neuron IDs]
    # }End of iteration

    log_file = open(inputFile, mode='r')  #'/home/adarsha/Research/Noxim/noxim-extended-for-neuromorphic/bin/carl-logs/8x8.log', mode='r')
    output_file = open(outputFilePath+'parse_result.txt', mode='w')

    log_file_lines = log_file.readlines()

    connection_table_file = open(outputFilePath+'connection_info.txt', mode='w')
    weight_table_file = open(outputFilePath+'weight_info.txt', mode='w')

    connectionTableParser(log_file_lines)

    connection_table_file.close()
    weight_table_file.close()

    log_file.close()
