/// @description  coment


// Batch
batchSize = 25;
batchPosition = 0;

action=[x,y]



room_speed=60

////////////////////////////примеры batch
// Просто предположим, что «примеры» — это массив, который содержит множество отдельных обучающих примеров.
index = 0;	// Индекс массива для примеров.
examples = [0,1,3];			// Вы можете построить, как хотите, свой список примеров.
examples[0] = {			// Не нужно хранить структуры.
	input : [0], 
	output : [0.3,0]		
};						// Поскольку функция активации "Tanh" может выводить только эти
						//Поэтому вам нужно нормализовать вывод, если они выходят за эти диапазоны.
examples[1] = {			
	input : [0.2], 
	output : [0.3,0]		
}
examples[2] = {			
	input : [0.8], 
	output : [0.3,0]		
}
var layers = [1,2,2,2];
mlpa = new mlp_array(layers);

error=0
sumError=1
batchError=1