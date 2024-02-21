// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://help.yoyogames.com/hc/en-us/articles/360005277377
 function memory() constructor //создание кортежа данных (ну типо буфера)
{
	enum type
	{
		next_values,//value по next_state. тут был log_prob
		log_prob,
		values,
		reward,
		done
	}
	
	next_values = []
	log_prob = []
	values = []
	reward = []
	done = []
	
	len_memory=0//длина памяти, можно узнать и array_lenght(rewards)

	static add = function(next_val,log_p,val,rew,don)
	{
		array_push(next_values,next_val)
		array_push(log_prob,log_p)
		array_push(values,val)
		array_push(reward,rew)
		array_push(done,don)
		len_memory+=1
		//show_message(values)
	}

	static clear = function()
	{
		array_clear(next_values)
		array_clear(log_prob)
		array_clear(values)
		array_clear(reward)
		array_clear(done)
		len_memory=0
	}

	static tuple = function()//Replay buffer -кортеж , ну типо все даные по порядку в массив
	{
		var array = []
		for (var i=0;i<array_length(next_values);i++)
		{
			var tupl = [next_values[i],log_prob[i],values[i],reward[i],done[i]]
			array_push(array,tupl)
		}
	return array_reverse(array)//я сделал специально в обратно порядке, написано что так нужно
	}
	
	static minibatch = function(tuple_array,numb)//это случайно выбранный набор обучающих примеров 
	{//numb - сколько в массиве наборов будет (из рандомных пазиций)
	 //Пример memory_array.minibatch(memory_array.tuple(),3)
		var array=[]
		for (var i=0;i<numb;i++)//сколько в масиве будет рандомных данных
		array[i]=tuple_array[irandom_range(0,array_length(tuple_array)-1)]
		
		return array
	}
}
/*
	//BATCH у тенора ханула
	// Повторить в том же кадре
	repeat(repeatTime) {//если больше чтобы для ускорения default 5
		// Возьмите случайные позиции для партии
		repeat(batchSize) {//рамер всех наборов в нутри
			i = irandom_range(0, array_length(objMulti.points)-1);
			input = [objMulti.points[i].x, objMulti.points[i].y];
			output = [objMulti.points[i].value];
			
			mlp.Forward(input);
			mlp.optimizer.Cost(MeanSquare, output);
			mlp.optimizer.Backward();
		}
	
		// Обновление весов на основе партии
		mlp.optimizer.Apply(learnRate);
	}
*/
 
function memoryPPO() constructor //создание кортежа данных (ну типо буфера)
{
	enum typePPO
	{
		next_values,//value по next_state. тут был log_prob
		log_prob,
		next_log_prob,
		values,
		reward,
		action,
		done
	}
	
	next_values = []
	log_prob = []
	next_log_prob = []
	values = []
	reward = []
	action = []
	done = []
	
	len_memory=0//длина памяти, можно узнать и array_lenght(rewards)

	static add = function(next_val,log_p,n_log_p,val,rew,act,don)
	{
		array_push(next_values,next_val)
		array_push(log_prob,log_p)
		array_push(next_log_prob,n_log_p)
		array_push(values,val)
		array_push(reward,rew)
		array_push(action,act)
		array_push(done,don)
		len_memory+=1
	}

	static clear = function()
	{
		array_clear(next_values)
		array_clear(log_prob)
		array_clear(next_log_prob)
		array_clear(values)
		array_clear(reward)
		array_clear(action)
		array_clear(done)
		len_memory=0
	}

	static tuple = function(gamma,discount_reward,normalize)//Replay buffer -кортеж , ну типо все даные по порядку в массив
	{	
		if discount_reward
		reward=calculate_returns(reward,gamma,normalize)
		
		var array = []
		for (var i=0;i<array_length(next_values);i++)
		{
			var tupl = [next_values[i],log_prob[i],next_log_prob[i],values[i],reward[i],action[i],done[i]]
			array_push(array,tupl)
		}
	return array
	}
	
	static minibatch = function(tuple_array,numb)//это случайно выбранный набор обучающих примеров 
	{//numb - сколько в массиве наборов будет (из рандомных пазиций)
	 //Пример memory_array.minibatch(memory_array.tuple(),3)
		var array=[]
		for (var i=0;i<numb;i++)//сколько в масиве будет рандомных данных
		array[i]=tuple_array[irandom_range(0,array_length(tuple_array)-1)]
		
		return array
	}
}