# Melhorando o Treinamento e a Topologia de um Decodificador de Canal baseado em Redes Neurais

-
Marcelo Vinícius Cysneiros Aragão, Samuel Baraldi Mafra e Felipe Augusto Pereira de Figueiredo (baseado em GRUBER, Tobias et al. [On deep learning-based channel decoding.](https://github.com/gruberto/DL-ChannelDecoding) In: 2017 51st Annual Conference on Information Sciences and Systems (CISS). IEEE, 2017. p. 1-6.)

### Executando as simulações

	docker pull tensorflow/tensorflow:latest-gpu
	docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash
	apt update && apt install git vim wget -y
	git clone https://github.com/marcelovca90/DL-ChannelDecoding.git
	cd DL-ChannelDecoding/scripts
	pip install matplotlib pandas
	./run.sh

##### Saída de exemplo:

	Model: "sequential_14"
	_________________________________________________________________
	 Layer (type)                Output Shape              Param #   
	=================================================================
	 modulator (Lambda)          (None, 16)                0         
	                                                                 
	 noise (Lambda)              (None, 16)                0         
	                                                                 
	 decoder_0 (Dense)           (None, 128)               2176      
	                                                                 
	 decoder_1 (Dense)           (None, 64)                8256      
	                                                                 
	 decoder_2 (Dense)           (None, 32)                2080      
	                                                                 
	 decoder_3 (Dense)           (None, 8)                 264       
	                                                                 
	=================================================================
	Total params: 12,776
	Trainable params: 12,776
	Non-trainable params: 0
	_________________________________________________________________
	2022-06-18 23:23:08.766694	random @ Adam @ Mep=2^18 fit started.
	2022-06-18 23:38:06.050502	random @ Adam @ Mep=2^18 fit finished (took 897.284 [s]).
	2022-06-18 23:38:06.742172	test @ sigmas=[1.         0.93162278 0.86324555 0.79486833 0.72649111 0.65811388
	 0.58973666 0.52135944 0.45298221 0.38460499 0.31622777]
	2022-06-18 23:38:06.747935	test @ sigmas(dB)=[ 0.          0.61519804  1.277313    1.99409613  2.77539396  3.63397895
	  4.58683749  5.65725523  6.87837702  8.29970172 10.        ]
	2022-06-18 23:39:20.703277	test @ sigma[0]=1.000	sigma_db[0]=0.000	nb_bits=16000000	nb_errors=3589326
	2022-06-18 23:40:34.717581	test @ sigma[1]=0.932	sigma_db[1]=0.615	nb_bits=16000000	nb_errors=3062144
	2022-06-18 23:41:48.939612	test @ sigma[2]=0.863	sigma_db[2]=1.277	nb_bits=16000000	nb_errors=2497056
	2022-06-18 23:43:05.342307	test @ sigma[3]=0.795	sigma_db[3]=1.994	nb_bits=16000000	nb_errors=1914599
	2022-06-18 23:44:19.313782	test @ sigma[4]=0.726	sigma_db[4]=2.775	nb_bits=16000000	nb_errors=1349638
	2022-06-18 23:45:33.081503	test @ sigma[5]=0.658	sigma_db[5]=3.634	nb_bits=16000000	nb_errors=843859
	2022-06-18 23:46:47.201872	test @ sigma[6]=0.590	sigma_db[6]=4.587	nb_bits=16000000	nb_errors=446177
	2022-06-18 23:48:01.259782	test @ sigma[7]=0.521	sigma_db[7]=5.657	nb_bits=16000000	nb_errors=185973
	2022-06-18 23:49:15.360117	test @ sigma[8]=0.453	sigma_db[8]=6.878	nb_bits=16000000	nb_errors=54946
	2022-06-18 23:50:29.356627	test @ sigma[9]=0.385	sigma_db[9]=8.300	nb_bits=16000000	nb_errors=9932
	2022-06-18 23:51:43.700041	test @ sigma[10]=0.316	sigma_db[10]=10.000	nb_bits=16000000	nb_errors=821

### Gerando as saídas e gráficos

	cd DL-ChannelDecoding/scripts
	python proc4paper-txt.py
	python proc4paper-json.py

##### Saída de exemplo:

![cpu_code=random_epochs=2^18_map](./experiments/scenario-9-sbrt-timed-16M/cpu_code=random_epochs=2^18_map.png "cpu_code=random_epochs=2^18_map")