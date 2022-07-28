from math import ceil

dim_mb = [[64, 16], [128, 32], [256, 64], [512, 128], [1024, 256]]
dim_model = [
	[1024, 64],
	[1024, 256, 64],
	[1024, 512, 256, 64],
	[1024, 512, 256, 128, 64]
]


for model in dim_model:
	for mb in dim_mb:
		print(len(model), mb[0], end = ' ')
		total_cluster = 0
		utilized_neuron = 0
		for i in range(len(model) - 1):
			if model[i] <= mb[0]:
				dc = ceil(model[i + 1] / mb[1])
				total_cluster += dc
				utilized_neuron += dc * model[i] + model[i + 1]
			else:
				num_unroll = model[i] // mb[0] * model[i + 1]
				dc = max(model[i] // mb[0], num_unroll // mb[1])
				total_cluster += dc
				utilized_neuron += dc * (mb[0] + min(mb[1], num_unroll // dc))
				if (num_unroll <= mb[0]):
					dc = ceil(model[i + 1] / mb[1])
					total_cluster += dc
					utilized_neuron += dc * num_unroll + model[i + 1]
				else:
					dc = max(num_unroll // mb[0], model[i + 1] // mb[1])
					total_cluster += dc
					utilized_neuron += num_unroll + model[i + 1]
		total_neuron = total_cluster * (mb[0] + mb[1])
		print(total_cluster, utilized_neuron / total_neuron)

