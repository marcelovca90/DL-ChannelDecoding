import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

base_folder = '../experiments/scenario-9-sbrt-timed-16M'

scenarios = [
    {'folder': base_folder + '/baseline', 'json': 'test.json', 'exponents': [10]},
    {'folder': base_folder + '/by-epoch', 'json': 'DL-CD-MAP-GPU-EP.json', 'exponents': [18, 20, 22]},
    {'folder': base_folder + '/by-topology-4-layers', 'json': 'DL-CD-MAP-GPU-4L.json', 'exponents': [20]},
    {'folder': base_folder + '/by-topology-5-layers', 'json': 'DL-CD-MAP-GPU-5L.json', 'exponents': [20]},
    {'folder': base_folder + '/by-topology-6-layers', 'json': 'DL-CD-MAP-GPU-6L.json', 'exponents': [20]}
]

codes = ['polar', 'random']
optimizers = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD']
SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 10

def json_to_csv(input_json, output_csv):
	with open(input_json, encoding='utf-8') as input_file:
		df = pd.read_json(input_file)
	df.to_csv(output_csv, encoding='utf-8', index=False)

for scenario in scenarios:

	os.makedirs(f"{scenario['folder']}/paper/csv", exist_ok=True)
	os.makedirs(f"{scenario['folder']}/paper/json", exist_ok=True)
	os.makedirs(f"{scenario['folder']}/paper/png", exist_ok=True)

	nb_epochs = [int(2**x) for x in scenario['exponents']]

	for code in codes:

		for nb_epoch in nb_epochs:
		
			nb_epoch_exp = int(np.log2(nb_epoch))

			fig = plt.figure(figsize=(11.69,8.27))

			# MAP results from digitized plots (CSV)
			df_map = pd.read_csv(f"{scenario['folder']}/map/{code}/digitized.csv")
			x_map = df_map['es_no_db']
			y_map = df_map['ber']
			plt.semilogy(x_map, y_map, label='MAP', linestyle='dotted', linewidth=2.0, color='black')

			data = {'x_MAP': x_map.tolist(), 'y_MAP': y_map.tolist()}

			for optimizer in optimizers:

				# NN results from simulation outputs (JSON)
				with open(f"{scenario['folder']}/{scenario['json']}", 'r') as file:
					df_nn = json.load(file)
				x_nn = df_nn[code][str(f'2^{nb_epoch_exp}')][optimizer]['x']
				y_nn = df_nn[code][str(f'2^{nb_epoch_exp}')][optimizer]['y']
				plt.semilogy(x_nn, y_nn, label=optimizer, linestyle='solid', linewidth=2.0)

				if 'x_NN' not in data:
					data[f'x_NN'] = x_nn
				data[f'y_{optimizer}'] = y_nn

				for i in range(len(x_map)-len(x_nn)):
					if len(data['x_NN']) != len(x_map):
						data['x_NN'].append(data['x_NN'][-1])
					data[f'y_{optimizer}'].append(data[f'y_{optimizer}'][-1])

			# persist JSON
			with open(f"{scenario['folder']}/paper/json/cpu_code={code}_epochs=2^{nb_epoch_exp}_map.json", 'w') as file:
				json.dump(obj=data, fp=file, indent=4)

			# persist CSV
			json_to_csv(
				f"{scenario['folder']}/paper/json/cpu_code={code}_epochs=2^{nb_epoch_exp}_map.json",
				f"{scenario['folder']}/paper/csv/cpu_code={code}_epochs=2^{nb_epoch_exp}_map.csv"
			)

			# persist PNG
			plt.grid(True, which='both')
			plt.legend(loc='lower left')
			plt.xlabel('$E_b/N_0$ [dB]')
			plt.ylabel('BER')
			plt.xlim(SNR_dB_start_Eb, SNR_dB_stop_Eb)
			plt.ylim(1e-5, 1e0)
			plt.title(code.capitalize() + ' code ($M_{ep}=2^{' + str(nb_epoch_exp) + '})$')
			plt.tight_layout(pad=0)
			plt.savefig(f"{scenario['folder']}/paper/png/cpu_code={code}_epochs=2^{nb_epoch_exp}_map.png")
