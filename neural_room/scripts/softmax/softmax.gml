// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://www.youtube.com/watch?v=KpKog-L9veg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=10
//это функция активации, фигня кароч
function softmax(array)
{
len=array_length(array)

// Применяем функцию softmax
var array_return;
var sum = 0;
for (var i = 0; i < len; i++) {
    array_return[i] = exp(array[i]);
    sum += array_return[i];
}

// Нормализуем значения
for (var i = 0; i < len; i++) {
    array_return[i] /= sum;
}
return array_return
}


function argmax(array)
{
	var maxIndex = 0;
	var return_array = array_create(array_length(array),0)
//так как мы взяли 0 индекс, в цикл добавляем 1 индекс, цикл начинаем с 1 индекса, так как 0 будет лишний цикл
	for (var i = 1; i < array_length(array); i++) {
	    if (array[i] > array[maxIndex]) 
	        maxIndex = i;
	}
	return_array[maxIndex]=1
	return return_array;
}


function ln_array(array_target)
{
	var old_array
	for (var i=0;i<array_length(array_target);i++)
	old_array[i]=ln(array_target[i]+0.000000001)
	return old_array
}

/// @func	CategorialCE(delta, predictions, targets);
/// @desc	Categorial Cross Entropy. Special case of cross-entropy.
/// @desc	Special as target-probability vector is 100% in one class.
/// @desc		-> Target should be one-hot vector (only one "1", others "0") -> eg. [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0];
/// @desc	Predictions should be probablity distribution -> sum up to 1. -> eg. [.7, .1, .2], [.0, .9, .1], [.3, .5, .2]
/// @desc	Because how predictions and targets should be defined, we can only care about "hot" prediction.
//function CategorialCE(delta, predictions, targets) {
//	//predictions = softmax
//	//targets = one-hot vector
//	var error = 0;
//	var iEnd = array_length(delta);
//	for(var i = 0; i < iEnd; i++) {
//		delta[@i] = 0;
//    }											// Normally: error += -target * ln(prediction)
//	// Find index for hot						//		
//	for(var i = 0; i < iEnd; i++) {				// But as targets array is type of [1, 0, 0, 0]
//		if (targets[i] == 1) {					// Wrong: error += -0 * ln(prediction)		-> error += 0;
//			error = (-1) * ln(predictions[i]);	// Correct: error += -1 * ln(prediction)	-> error += -ln(prediction)
//			delta[@i] = (-1) / predictions[i];	// This can be simplified!
//			break;								// Just search "hot" index.
//		}
//    }
//    return error;
//}



function random_choice() 
{
	//https://www.gmlscripts.com/script/random_weighted
    var sum = 0;
    for (var i=0; i<argument_count; i++) {
        sum += argument[i];
    }
    var rnd = random(sum);
    for (var i=0; i<argument_count; i++) {
        if (rnd < argument[i]) return i;
        rnd -= argument[i];
    }
}


/// @param probs Массив вероятностей действий
// Функция для сэмплирования действий
function sample_categorical(probs) {
    var rand_vals = random_multinomial(probs);
	var max_val = -1;
    var max_index = -1;
show_message(rand_vals)
    for (var i = 0; i < array_length(rand_vals); i++) {
        if (rand_vals[i] > max_val) {
            max_val = rand_vals[i];
            max_index = i;
        }
    }

    return max_index;
}

function random_multinomial(probabilities)
{//по сути как random_choice , только сделаный нейроннкой
	// probabilities  Массив вероятностей
	// num_samples  Число выборок

	var total = 0;
	var array_hot_one=[]
	repeat(array_length(probabilities))
	array_push(array_hot_one,0)
	// Подсчитываем общую сумму вероятностей
	for (var i = 0; i < array_length(probabilities); i++) {
	    total += probabilities[i];
	}

	// Генерируем случайное число в диапазоне от 0 до общей суммы вероятностей
	var rand = random(total);
	
	
	// Выбираем действие на основе вероятностей
	var accum = 0;
	for (var i = 0; i < array_length(probabilities); i++) {
	    accum += probabilities[i];
	    if (rand < accum) {
			//return i;  // Возвращаем выбранное действие
			array_hot_one[i]=1// Возвращаем выбранное действие
			return array_hot_one
	    }
	}

	show_message("PIZDEC")
	return array_hot_one;  //-1 В случае ошибки
}





/// @func	MeanSquare(delta, predictions, targets);
/// @desc	Mean Squared Error (MSE).
//Среднеквадратическая ошибка 
/// @desc	Довольно универсальная функция затрат (cost), но лучше работает с регрессионными задачами.
function MeanSquare(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
		error += sqr(prediction - target);
		delta[@i] = (prediction - target);
	}
    return .5 * error / iEnd;
}

function array_max(_array) {
	var _min = _array[0];
	for(var i = 1; i < array_length(_array); i++){
		_min = max(_array[i], _min);
	}
	return _min;
}


function entropy(array,beta=0.0000001)
{	
	var old_var=0
	for(var i=0;i<array_length(array);i++)
		old_var+=(-array[i]*ln(array[i]+beta))//log2

	return old_var
}


//	SIGMOID
//________________________________________________________________________________________________________________
/// @func	Sigmoid(input);
/// @desc	Non-linear activation function. Creates S-curve similiar to Tanh. Return value is always between 0 and +1.
/// @param	{real}	input
function Sigmoid(input) {
	return (1 / (1 + exp(-input)));
}
/// @func	SigmoidDerivative(input);
/// @desc	Derivative of Sigmoid-function
/// @param	{real}	input
function SigmoidDerivative(input) {
	input = Sigmoid(input);	// To call Sigmoid only once.
	return (input * (1 - input));
}

///@param _array [1,2,3,3]
function array_mean(__array) {
	
	var __arrayValue = 0;
	var __arrayLength = array_length(__array);
	for (var i = 0; i < __arrayLength; i++) {
		__arrayValue += __array[i];
	}
	return __arrayValue / __arrayLength;
}

function subarray_mean(array) {
	//главное чтобы подмассивы были одинаковыми 
	//Пример subarray_mean([[1,2,1],[2,5,1],[8,5,3],[1,22,3]])
	var arrays = array;
	var result = array_create(array_length(arrays[0]), 0);

	for (var i = 0; i < array_length(arrays); i++) {
	    for (var j = 0; j < array_length(arrays[0]); j++) {
	        result[j] += arrays[i][j];
	    }
	}

	for (var j = 0; j < array_length(result); j++) {
	    result[j] /= array_length(arrays);
	}

	return result;
}


function array_clear(array) {
	return array_delete(array, 0, array_length(array));
}

function cal_reward(obj1, obj2)
{
	var reward=-1
    var distance = point_distance(obj1.x, obj1.y, obj2.x, obj2.y);
	if distance<20
	reward=100
	//return reward+1/distance;
	//reward=-1
   // return (10 / reward)*10;
   //show_message(lerp(room_height,0, clamp(distance/room_height,0,1) )/room_height)
	return lerp(room_height/2,0, clamp(distance/room_height/2,0,1) )/room_height;
}
///дисконтированное вознаграждение (кароче награда со скидкой)
/// @param rewards Список наград
/// @param discount_factor Коэффициент дисконтирования
/// @param normalize Нормализовать результаты или нет
function calculate_returns(rewards, discount_factor, normalize=true)
{

	var returns = [];
	var R = 0;

	for (var i = array_length(rewards) - 1; i >= 0; i--)
	{
	    R = rewards[i] + R * discount_factor;
	    array_insert(returns, 0, R);
	}

	if (normalize)
	{
	    var m = array_mean(returns);
	    var std_dev = array_standard_deviation(returns);

	    for (var i = 0; i < array_length(returns); i++)
	    {
	        returns[i] = (returns[i] - m) / std_dev;
	    }
	}

	return returns;
}

function calculate_advantages(returns, values, normalize=true)
{
	//Пример calculate_advantages(reward, values, normalize=true)  https://colab.research.google.com/drive/1ReX2ymCK2NIrSYxLvJFrY3GhNxet1FRU#scrollTo=WK4Otv0RfLOi
	/// @param returns Массив возвращенных значений
	/// @param values Массив оценок
	/// @param normalize Нормализовать результаты или нет
	var advantages = array_create(array_length(returns), 1);
	
	// Вычисляем преимущества
	for (var i = 0; i < array_length(returns); i++)
	{

	    advantages[i] = returns[i] - values[i]; 
		//show_message($"{advantages}={values} - {returns}")
	}

	if (normalize)
	{
	    var m = array_mean(advantages);
	    var std_dev = array_standard_deviation(advantages);
		
//show_message($"{advantages} m {m} / {std_dev}")
	    for (var i = 0; i < array_length(advantages); i++)
	    {
			
	        advantages[i] = (advantages[i] - m) / std_dev;
			
	    }
		
	}
	return advantages;
}



/// улучшенный calculate_advantage
/// @param advantages Массив преимуществ (calculate_advantage)
/// @param gamma Коэффициент дисконтирования
/// @param lambda Параметр GAE
function gae(advantages, gamma, lambda)
{
	var num_steps = array_length(advantages);
	var gae_advantages = array_create(num_steps, 0);

	for (var t = 0; t < num_steps; t++) {
	    var delta_sum = 0;

	    for (var l = 0; l < num_steps - t; l++) {
	        var delta = power(gamma * lambda,l) * advantages[t];//var delta = advantages[t] + power(gamma * lambda,l) * (advantages[t] - gamma * lambda * advantages[t]);
	        gae_advantages[t] += delta;
	    }
	}
	return gae_advantages
}

///вычисление сети критик
//для упрошения
function compute_critic_value(array,name_critic_network)
{
	name_critic_network.Forward(array)
	return name_critic_network.Output()[0]
}

/*
function entropy(action_probs,entropy_coef)
{
	var entropy_bonus = 0;
	var num_actions = array_length(action_probs);

	for (var i = 0; i < num_actions; i++) 
	    entropy_bonus += -action_probs[i] * log2(action_probs[i] + 0.00001); // Добавляем небольшой эпсилон для избежания логарифма от нуля
	
	entropy_bonus=(entropy_bonus / num_actions)*entropy_coef // Усредняем
	return entropy_bonus
}
*/