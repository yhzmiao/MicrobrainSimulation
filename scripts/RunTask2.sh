for((i=2;i<=5;i++))
do
    ./MicrobrainSimulation -t 2 -m Dummy_$i
done
./MicrobrainSimulation -t 2 -m CIFAR10
