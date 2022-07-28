for((i=100;i<=1000;i+=100))
do
    #./MicrobrainSimulation -t 1 -r $i -d MNIST_16 -m MNIST_negative -l 1000
    ./MicrobrainSimulation -t 1 -r $i -d MNIST_32 -m MNIST_largescale_2 -l 1000
    #./MicrobrainSimulation -t 1 -r $i -d MNIST_32 -m MNIST_largescale_3 -l 10000
    #./MicrobrainSimulation -t 1 -r $i -d MNIST_32 -m MNIST_largescale_3_2 -l 1000
    #./MicrobrainSimulation -t 1 -r $i -d MNIST_32 -m MNIST_largescale_4 -l 1000
done
