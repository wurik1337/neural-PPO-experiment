#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________
Как использовать
Создать: mlp = новый mlp_array(layers); Задайте параметры, как указано в конструкторе.

Получить вывод mlp.Forward(массив); Ввод подается как массив. Размер массива должен быть того же размера, что и входной слой. (хотя ввод зажат).
Получить вывод mlp.Output(); Возвращает вывод последнего Forward().

Поезд MLP 1. mlp.Forward(input);
Сначала обновите вывод MLP с помощью примера-ввода, затем используйте обратное распространение с примером-выводом.
2. млп.Стоимость(цель); Используйте функцию стоимости. Это обновленная дельта ошибки вывода.
3. млп.Назад(); Обратно распространяет дельту ошибки выходного слоя через MLP, обновляя дельты и градиенты.
4. млп.Применить(.03);
Стохастический градиентный спуск. При этом используются градиенты, созданные обратным распространением, вычисляется их среднее значение.
																
Обучите MLP с пакетами: повторите шаги 1-3 с другими входными/целевыми парами. Размер партии подразумевает, сколько раз вы повторяете этот процесс.
Повторение этого процесса приводит к накоплению дельт и градиентов и еще не приводит к обновлению весов и смещений.
Количество повторов может быть произвольным,
средние градиенты и дельты рассчитываются автоматически при использовании «Apply()».
Используйте «Применить()», чтобы использовать накопленные градиенты и дельты для обновления весов и смещений.
Градиенты и дельты сбрасываются и готовы к накоплению следующей партии.
Повторяйте это до тех пор, пока ошибка не достигнет желаемого уровня.
																
		Destroy:		Так как это struct+arrays, он автоматически собирается мусором, когда на него больше не ссылаются.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_array(layerSizes);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @param	{array}		layerSizes		Give layer-sizes as array of integers.		Like:	[10, 32, 16, 4]
function mlp_array(layerSizeArray) constructor {
	
	#region /// INITIALIZE NETWORK - Neurons, weights and biases.
		// Configure layers.
		session	= 0;															// Сколько тренировочных сессий было до Apply(). Это нужно для того, чтобы разделить градиенты, чтобы получить среднее значение нескольких примеров. Это сделано для того, чтобы пользователь мог вычислять примеры с течением времени.
		layerCount	= array_length(layerSizeArray);								// Количество слоев, определяемое при создании структуры
		layerSizes	= array_create(layerCount);									// Кэшируем размеры слоев. Тоже выглядит чище.
		array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);
															
		// Initialize neurons + their deltas.									
		activity	= [[undefined]];											// Объединенные выходные сигналы от предыдущих слоев, умноженные на связанный вес. Нужно сохранить для обучения.
		output		= [[undefined]];											// Выходной сигнал: активация (активность + смещение).
		bias		= [[undefined]];											// Смещение активности нейронов
		delta		= [[undefined]];											// Разница между фактическим выходом и желаемым результатом. Рассчитано из текущего примера
		deltaAdd	= [[undefined]];											// Additivie/Cumulative delta от нескольких примеров
																			
		var i, j, k;															
		for(i = 0; i < layerCount;	i++) {	//4 слоя									// Инициализируйте начальные значения.
		for(j = 0; j < layerSizes[i]; j++) {	//берёт числы из масивов по слойно (сначало 1 это4  слой потом 2 это 32 и тп)								
			activity[i][j]	= 0;												
			output[i][j]	= 0;												
			bias[i]		= random_range(-.2,+.2);							// Give default random bias.
			delta[i][j]		= 0;												
			deltaAdd[i][j]	= 0;												
		}}																		
																	
		// Инициализировать веса и градиенты									
		weights		= [[[undefined]]];											// Связь между нейронами
		gradients	= [[[undefined]]];											// Градиент используется в обучении для обновления весов. 
		for(i = 1; i < layerCount; i++) {	    //1 2 3									// Инициализировать начальные значения
		for(j = 0; j < layerSizes[i]; j++) {	//3 1 1								
		for(k = 0; k < layerSizes[i-1]; k++) {	//1 3 1								
			weights[i][j][k] = random_range(-.5,+.5);							// Дайте случайный вес по умолчанию.
			gradients[i][j][k] = 0;		
		}}}
		

	#endregion

	/// @func	Output();
	/// @desc	Возвращает выходной массив из MLP, не обновляется
	/// @return	{array}
	static Output = function() {
		return output[layerCount-1]; //-1 потому что масив начинается с 0 а не 1
	}
	
		
	/// @func	Forward(input);
	/// @desc	Обновляет выходные данные MLP по заданному входному массиву. например: млп.Вперед([1,0,0,1]);
	/// @param	{array}		input		Введите входные данные в виде массива.
	static Forward = function(inputArray) {

		var i, j, k, minSize;
				
		// Установить ввод. Поместите выходные данные нейронов первого слоя в качестве заданных входных данных
		i = 0;
		minSize = min(layerSizes[i], array_length(inputArray));
		array_copy(output[i], 0, inputArray, 0, minSize);//утсанавливает на input  значение аргумента0
		
		// обновление данных ну или слоёв
		//ВАЖНО!!! цикл идентичный, когда мы создавали веса и градиент(выше есть)
		for(i = 1; i < layerCount; i++) {//по слоям
		for(j = 0; j < layerSizes[i]; j++) {//по нейронам в слою
			activity[i][j] = 0;													// Сбросить активность. Он сохраняется между раундами для обратного распространения
			for(k = 0; k < layerSizes[i-1]; k++) {	//по связям с этим нейронам							
				activity[i][j] += output[i-1][k] * weights[i][j][k];			//Активность представляет собой сумму всех направленных сигналов предыдущего слоя. Выходной сигнал * вес, вес - это связь между двумя нейронами.
			}																	
			output[i][j] = Tanh(activity[i][j] + bias[i]);//tanh()					// Результатом является (активность + смещение), но также выполняется функция активации, которая должна быть нелинейной.
		}}
		
		// Return output. 
		return Output();
	}
		
	//Желаемый результат
	/// @func	Cost(targetOutput);
	/// @desc	Функция потерь. Использует среднеквадратичную ошибку.
	/// @desc	Вычисляет ошибку и обновляет дельту ошибки выходного слоя.
	/// @param	{array}		targetOutput	пример вывода, желаемый результат данного ввода
	/////Важно!!!! Дельта тут как изменение или раница (пример: Выход=0, target=3, delta=0-3)!!!!!
	static Cost = function(target) {
		var i = layerCount - 1;//если 4 слоя то будет 3
		// Вычислите дельту, используйте производную MSE.
		//я это написал чтобы небыло цифр в delta (они меня напрегают)
		//for(var j = 0; j < layerSizes[0]; j++) {
		//	delta[0][j] = 0;
		//}
		
		//по сути мы узнаём разницу между выходом сети и желаемым результатом(дельта) последних нейронов
		for(var j = 0; j < layerSizes[i]; j++) {//по нейронам в последнем слое
			delta[i][j] = (output[i][j] - target[j]);//output(предсказание)-желаймый результат =(и это будет дельта последнего слоя, по всем нейронам в нём)
		}

		// Рассчитать ошибку, использовать MSE
		var error = 0;
		for(var j = 0; j < layerSizes[i]; j++) {//повторяем хуйню что выше
			error += sqr(output[i][j] - target[j]);//сумма разниц выходов(предсказание) и желаемых резулт в степени 2
		}
		
		// Возвращает среднюю ошибку. (2 здесь для технических целей, поэтому производная верна.)
		return error / (layerSizes[i] * 2);
		
		
	}
	
		function CrossEntropy(predictions_softmax, targets_one_hot,advantage) {
		//predictions = softmax (array)
		//targets = one-hot vector (array)
		var i = layerCount - 1;//если 4 слоя то будет 3

		var error = 0;
		for(var j = 0; j < layerSizes[i]; j++) {//по нейронам в последнем слое
			delta[i][j] = 0;
	    }
		delta[0][0]=0
		delta[0][1]=0
		// Normally: error += -target * ln(prediction)
		// Find index for hot						//		
		for(var j = 0; j < array_length(targets_one_hot); j++) {//default for(var j = 0; j < layerSizes[i]; j++)	// But as targets array is type of [1, 0, 0, 0]
			for (var w=0;w<array_length(targets_one_hot[j]);w++)
			{
				// Wrong: error += -0 * ln(prediction)		-> error += 0;
					error = 1//(-1)*targets_one_hot[j][w] * ln(predictions_softmax[j][w]);	//default error = (-1) * ln(predictions_softmax[j])///было и так error = (-1) * ln(predictions_softmax[targets_one_hot[j]])*advantage[0];; /// Correct: error += -1 * ln(prediction)	-> error += -ln(prediction)
					//show_message(predictions_softmax[j][w])
					//show_message(ln(predictions_softmax[j][w]))
					delta[i][w]=(-1)*targets_one_hot[j][w] / predictions_softmax[j][w];//*advantage[j]//default delta[i][w]=(-1)*targets_one_hot[j][w] / predictions_softmax[j][w];	// This can be simplified!
					//show_message(predictions_softmax[j][w])
					
			}

	    }
	    return error;
	}
	
	
	/// @func	CategorialCE(delta, predictions, targets);
	/// @desc	Categorial Cross Entropy. Special case of cross-entropy.
	/// @desc	Special as target-probability vector is 100% in one class.
	/// @desc		-> Target should be one-hot vector (only one "1", others "0") -> eg. [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0];
	/// @desc	Predictions should be probablity distribution -> sum up to 1. -> eg. [.7, .1, .2], [.0, .9, .1], [.3, .5, .2]
	/// @desc	Because how predictions and targets should be defined, we can only care about "hot" prediction.
	function Cost_CategorialCE(predictions_softmax, targets_one_hot,advantage) {
		//predictions = softmax (array)
		//targets = one-hot vector (array)
		var i = layerCount - 1;//если 4 слоя то будет 3

		var error = 0;
		for(var j = 0; j < layerSizes[i]; j++) {//по нейронам в последнем слое
			delta[i][j] = 0;
	    }

		delta[0][0]=0
		delta[0][1]=0
		// Normally: error += -target * ln(prediction)
		// Find index for hot						//		
		for(var j = 0; j < array_length(targets_one_hot); j++) {//default for(var j = 0; j < layerSizes[i]; j++)	// But as targets array is type of [1, 0, 0, 0]
			for (var w=0;w<array_length(targets_one_hot[j]);w++)
			{

				if (targets_one_hot[j][w] == 1) {					// Wrong: error += -0 * ln(prediction)		-> error += 0;
					error = (-1) * ln(predictions_softmax[j][w]);	//default error = (-1) * ln(predictions_softmax[j])///было и так error = (-1) * ln(predictions_softmax[targets_one_hot[j]])*advantage[0];; /// Correct: error += -1 * ln(prediction)	-> error += -ln(prediction)
					delta[i][w]= output[i][w]-(advantage[j]) // / predictions_softmax[j][w];//*advantage[j]//default delta[i][j] = (-1) / predictions_softmax[j]///было и так delta[i][w] = (-1) / predictions_softmax[j][w]*advantage[j]	// This can be simplified!
					//show_message(predictions_softmax[j][w])
					if keyboard_check(vk_control)
					{
						show_message(advantage)
						show_message(output[i][w]-(advantage[j]*(-ln(predictions_softmax[j][w]))) ) //output[i][w]-(advantage[j])
					}
					//delta[i][w] =(-1) / (predictions_softmax[j][w])//оригинал строки
					//break;								// Just search "hot" index.
				}
				//else
				//delta[i][w]= output[i][w]-(-advantage[j])
			}

	    }
	    return error;
	}
	
	
	/// @param target_probs Массив вероятностей новой стратегии
	/// @param old_probs Массив вероятностей предыдущей стратегии
	/// @param advantages Массив преимуществ (advantages)
	/// @param epsilon Параметр ограничения изменения стратегии (clip)
	function ppo_loss(target_probs, old_probs, advantages,one_hot, epsilon)
	{

		var i = layerCount - 1;
		var num_bath = array_length(target_probs);
		var num_action = array_length(target_probs[0]);
		var loss = 0;
		

for (var d=0;d<array_length(delta[0]);d++)
delta[0][d]=0
	
		
		for(var j = 0; j < num_bath; j++) {//default for(var j = 0; j < layerSizes[i]; j++)	// But as targets array is type of [1, 0, 0, 0]
			for (var w=0; w<num_action;w++)
			{
				if (one_hot[j][w] == 1)
				{
				    var ratio = target_probs[j][w] / old_probs[j][w];
				    var clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon);//ограничитель
				//show_message($"target_probs{target_probs}old_probs{old_probs}advantages{advantages} targets_one_hot {one_hot}")
		
				    var surrogate_loss = -min(ratio * advantages[j], clipped_ratio * advantages[j]);
				    loss += surrogate_loss;
					//var s=sign(surrogate_loss)
					delta[i][w]=surrogate_loss
				}
				else
			delta[i][w]= output[i][w]
			//show_message(delta)///я на этом моменте остановился, я хз что делать с дельта,one_hot я добавил ради дельта но хз нужен или нет.
			// Вычисление градиентов (дельта)
	        //if (ratio * advantages[i] > surrogate_loss) {
	        //    delta[j][one_hot[i]] += surrogate_loss / old_probs[i];
	        //} else {
	        //    delta[j][one_hot[i]] += ratio * advantages[i] / old_probs[i];
	        //}
			//delta[j][one_hot[i]] += surrogate_loss
			}
		//show_message(delta)
		}
		return loss / w;
	}
	
	

	/// @func	Backward();
	/// @desc	Backpropagates output-error towards input 
	static Backward = function() {//РАБОТАЕТ СПРАВА НА ЛЕВА!!!!!!!!!!!!!!
		var i, j, k;													
		
		// Backpropagate through layers. From last to first.
		var varDeltaActivity;
		for(i = layerCount - 1; i > 0; i--) {// по слоям с последнего по первый !!!!!!ОБРАТНЫЙ ЦИКЛ!!!!!
		for(j = 0; j < layerSizes[i]; j++) { //по нейронам в слоях
			//ниже переменная varDeltaActivity умнажается на все связи с этим нейроном
			varDeltaActivity = delta[i][j] * TanhDerivative(activity[i][j]);//tanhderivative//разность(правел. и нужн. ответ.) * скорость функции(от положения активации)		// Кэшировать это. Функция активации и ее производная могут быть дорогостоящими расчетами.
			for(k = 0; k < layerSizes[i-1]; k++) {//по связям
				gradients[i][j][k] += output[i-1][k] * varDeltaActivity;			// Градиенты суммируются, суммируются. В Apply() они делятся для получения среднего значения.
				delta[i-1][k] += weights[i][j][k] * varDeltaActivity;				// Обновите дельту для связанного слоя. Это будет использоваться, когда мы перейдем к предыдущему слою.
			}																		
			deltaAdd[i][j] += varDeltaActivity;										//Суммарная дельта
			delta[i][j] = 0;														// Reset for next example.
		}}																			
		session++;	//нужна эта переменная , если apply не сразу применён, а через какоето время	// Это была 1 тренировка, добавьте в счет. В «Применить» это используется для разделения дельт и градиентов для получения среднего значения.								
	//show_message(delta)
	}
	

	/// @func	Apply(learnRate); узнать скорость
	/// @desc	Stochastic Gradient Descent.  Стохастический градиентный спуск.
	/// @desc	Обновляет веса и смещения в соответствии с градиентами/дельта. Они рассчитываются в обратном распространении
	/// @param	{real}	learnRate	Насколько быстро MLP обучается (обычно от 0,1 до 0,001).
	///								Если скорость обучения слишком высока, она может перепрыгнуть через локальный минимум. Это снова может привести к тому, что MLP продолжит перепрыгивать через него с другого направления.


	static Apply = function(learnRate,Adamm=false) {//РАБОТАЕТ СЛЕВА НА ПРАВА	
		if Adamm=false
		{//stohastic gradient
		var i, j, k;
			//Разделите на количество тренировок, чтобы получить среднее значение для нескольких тренировок. В противном случае градиенты - это просто их сумма.
			learnRate = learnRate / session;	//нужна эта переменная , если apply не сразу применён, а через какоето время														
			// Update weights & biases												
			for(i = 1; i < layerCount; i++) {	//по слоям									
			for(j = 0; j < layerSizes[i]; j++) {	//по нейронам								
				bias[i] += -learnRate * deltaAdd[i][j];							// Мы хотим обновлять в направлении, противоположном дельтам и градиентам, поэтому минус.
				deltaAdd[i][j] = 0; //сбрасываем разницу
				for(var k = 0; k < layerSizes[i-1]; k++) {//по связям с предыдущими нейронами(в прошлом слою)
					weights[i][j][k] += -learnRate * gradients[i][j][k];
					gradients[i][j][k] = 0; //сбрасываем градиент
				}
			
			}}
			// Counting starts again. Счет начинается снова.
			session = 0;
		}
		else
		{//ADAM

		betaOne = .9;
		betaTwo = .999;
		iteration = 1;
		oneMoment = array_create(layerCount, undefined);
		twoMoment = array_create(layerCount, undefined);
		var i, j, jEnd, kEnd;
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
			oneMoment[@i] = array_create(jEnd, undefined);
			twoMoment[@i] = array_create(jEnd, undefined);
		for(j = 0; j < jEnd; j++) {
			oneMoment[@i][@j] = array_create(kEnd, 0);
			twoMoment[@i][@j] = array_create(kEnd, 0);
		}}
	
		//static Apply = function(learnRate) {
			var i, j, k, jEnd, kEnd;
			var gradient, unbiasOne, unbiasTwo;
			var epsilon = .001;	// Чтобы не делить на 0.
			var divider = 1 / session;						// Кэширование.
			var oneDivider = 1 / (1 - power(betaOne, iteration));	// 
			var twoDivider = 1 / (1 - power(betaTwo, iteration));	// 
		
			// Обновите веса и смещения.
			for(i = 1; i < layerCount; i++) {
				jEnd = layerSizes[i];
				kEnd = layerSizes[i-1];
			for(j = 0; j < jEnd; j++) {
				bias[@i] += -learnRate * deltaAdd[i][j] * divider;//немного тут переделал
				deltaAdd[@i][@j] = 0;
			for(k = 0; k < kEnd; k++) {
				// Рассчитать импульс (возможно как в Momentum)
				gradient = gradients[i][j][k] * divider;
				gradients[@i][@j][@k] = 0;
				oneMoment[@i][@j][@k] = betaOne * oneMoment[i][j][k] + (1 - betaOne) * gradient;
				twoMoment[@i][@j][@k] = betaTwo * twoMoment[i][j][k] + (1 - betaTwo) * gradient * gradient;
				// Коррекция смещения
				unbiasOne = oneMoment[i][j][k] * oneDivider;
				unbiasTwo = twoMoment[i][j][k] * twoDivider;
				// AdaGrad.
				weights[@i][@j][@k] += -learnRate * unbiasOne / (sqrt(unbiasTwo) + epsilon);
			}}}
				
			session = 0;	// Счет начинается снова. так то он увеличивается bacward
			iteration++;		// Сделано на одну итерацию Адама больше
		//}		
		}
	}




	
	/// @func	Destroy();
	static Destroy = function() {
		// Destroy datastructures.
		activity = undefined;
		output = undefined;
		bias = undefined;
		delta = undefined;
		deltaAdd = undefined;
		weights = undefined;
		gradients = undefined;
		
		// Storing no layers.
		layerCount = 0;
		layerSizes = [0];	
	}
}








