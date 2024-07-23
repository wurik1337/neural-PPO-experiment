randomize()


// Гиперпараметры PPO
gamma = 0.99; // Коэффициент дисконтирования
lambda = 0.95; // Коэффициент для GAE
clip_ratio = 0.2; // Коэффициент для ограничения изменений
policy_learning_rate = 0.1;//0.0003
value_learning_rate = 0.1;//0.001
batch_size = 20; // Размер минибатча
epochs = 1; // Количество эпох для обновления

// Инициализация дополнительных переменных
training = false;
total_reward = 0;
timestep = 0;
max_timesteps = 20; // Количество шагов до обновления




function mlp_array(layerSizeArray,activationFunctionsArray) constructor {
    #region /// INITIALIZE NETWORK - Neurons, weights and biases.
    session = 0;
    layerCount = array_length(layerSizeArray);//число слоёв
    layerSizes = array_create(layerCount);//число нейроннов в слое
    array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);

    // Initialize neurons + their deltas.
    activity = []; //активации нейронов
    output = [];
    bias = []; //нейрон смещения
    delta = []; // Разница между фактическим выходом и желаемым результатом. Рассчитано из текущего примера
    //deltaAdd = [];
	timestep = 0; //служит для усиления моментов в ОПТИМИЗАТОРАХ (SGD, Adam ....)

    for (var i = 0; i < layerCount; i++) {
        activity[i] = array_create(layerSizes[i]);
        output[i] = array_create(layerSizes[i]);
        delta[i] = array_create(layerSizes[i]);
        //deltaAdd[i] = array_create(layerSizes[i]);
		bias[i]		= random_range(-.2,+.2);
    }

    // Initialize weights and gradients
    weights = [];
    gradients = [];
    for (var i = 1; i < layerCount; i++) {
        weights[i] = [];
        gradients[i] = [];
        for (var j = 0; j < layerSizes[i]; j++) {
            weights[i][j] = array_create(layerSizes[i - 1]);
            gradients[i][j] = array_create(layerSizes[i - 1]);
            for (var k = 0; k < layerSizes[i - 1]; k++) {
                weights[i][j][k] = random_range(-.5, +.5); // Give default random weight.
            }
        }
    }
	
	    // Initialize activation functions
    if (array_length(activationFunctionsArray) != layerCount - 1) {
        show_error("Количество функций активации должно соответствовать количеству слоев, за исключением входного слоя.", true);
    }
	
    activationFunctions = array_create(layerCount - 1);
    activationFunctionsDerivative = array_create(layerCount - 1);
    array_copy(activationFunctions, 0, activationFunctionsArray, 0, layerCount - 1);

    // Заполняем соответствующими производными
    for (var i = 0; i < layerCount - 1; i++) {
        switch(activationFunctions[i]) {
            case Tanh:
                activationFunctionsDerivative[i] = TanhDerivative;
                break;
            case Relu:
                activationFunctionsDerivative[i] = ReluDerivative;
                break;
            case Sigmoid:
                activationFunctionsDerivative[i] = SigmoidDerivative;
                break;
            default:
                show_error("Неизвестная функция активации.", true);
        }
    }
    #endregion

    /// @func Forward
    /// @desc Выполняет прямое распространение через сеть
    /// @param {array} inputData - входные данные для сети
    /// @return {array} - выходные данные сети
    static Forward = function (inputData) {
        // Проверяем, совпадает ли размер входных данных с размером первого слоя
        if (array_length(inputData) != layerSizes[0]) {
            show_error("Размер входных данных не совпадает с размером первого слоя", true);
        }

        // Устанавливаем входные данные как выходные данные первого слоя
        array_copy(output[0], 0, inputData, 0, array_length(inputData));

        // Прямое распространение через каждый слой
        for (var i = 1; i < layerCount; i++) {
            for (var j = 0; j < layerSizes[i]; j++) {
                for (var k = 0; k < layerSizes[i - 1]; k++) {
                    activity[i][j] = output[i - 1][k] * weights[i][j][k]; // Добавляем взвешенные входные данные
                }
				// Применяем соответствующую функцию активации к сумме активностей
                output[i][j] = activationFunctions[i - 1](activity[i][j]+bias[i]);
                //output[i][j] = Tanh(activity[i][j]+bias[i]);
            }
        }
        // Возвращаем выходной массив последнего слоя
        return output[layerCount - 1];
    }

    static Output = function() {
        return activity[layerCount - 1];
    }



/// @func LossFunction
/// @desc Вычисляет дельты для выходного слоя на основе производной функции потерь
/// @param {array} error - массив ошибок, возвращаемый производной функции потерь
/// @return {void}
static LossFunction = function(error) {
    var outputLayerIndex = layerCount - 1;

    for (var j = 0; j < layerSizes[outputLayerIndex]; j++) {
        delta[outputLayerIndex][j] = error[j] * activationFunctionsDerivative[outputLayerIndex - 1](activity[outputLayerIndex][j]); // Используем правильную производную функции активации
	}
}

/// @func Backpropagate
/// @desc Выполняет обратное распространение ошибки через сеть
/// @param {function} updateWeights - метод обновления весов
/// @param {real} learningRate - скорость обучения
/// @return {void}
static Backpropagate = function(updateWeights, learningRate) {
    // Обратное распространение через скрытые слои
    for (var i = layerCount - 2; i >= 0; i--) {
        for (var j = 0; j < layerSizes[i]; j++) {
            var errorSum = 0;
            for (var k = 0; k < layerSizes[i + 1]; k++) {
                errorSum += delta[i + 1][k] * weights[i + 1][k][j];
            }
            delta[i][j] = errorSum * activationFunctionsDerivative[i](activity[i][j]); // Используем правильную производную функции активации						
        }
    }

    // Обновляем веса и смещения
    updateWeights(learningRate);
}

    /// @func SGDUpdateWeights
    /// @desc Метод обновления весов для SGD
    /// @param {real} learningRate - скорость обучения
    /// @return {void}
    static SGDUpdateWeights = function(learningRate) {
        for (var i = 1; i < layerCount; i++) {
            for (var j = 0; j < layerSizes[i]; j++) {
                for (var k = 0; k < layerSizes[i - 1]; k++) {
                    weights[i][j][k] -= learningRate * delta[i][j] * output[i - 1][k];
                }
                bias[i] -= learningRate * delta[i][j];
            }
        }
    }

	/// @func AdamUpdateWeights
	/// @desc Метод обновления весов для Adam с внутренней инициализацией параметров
	/// @param {real} learningRate - скорость обучения
	/// @param {real} beta1 - коэффициент для моментов первого порядка
	/// @param {real} beta2 - коэффициент для моментов второго порядка
	/// @param {real} epsilon - малое значение для числовой стабильности
	/// @return {void}
	static AdamUpdateWeights = function(learningRate) {
	    // Инициализация параметров

	        var m_t = []; //Моменты первого порядка
	        var v_t = []; //Моменты второго порядка
		
			var epsilon = .001;	// Чтобы не делить на 0.
			var beta1 = .9; //коэффициент для моментов первого порядка
			var beta2 = .999; //коэффициент для моментов второго порядка
	        // Инициализация m_t и v_t
	        for (var i = 1; i < layerCount; i++) {
	            m_t[i] = []; //Моменты первого порядка
	            v_t[i] = []; //Моменты второго порядка
	            for (var j = 0; j < layerSizes[i]; j++) {
	                m_t[i][j] = array_create(layerSizes[i - 1]);
	                v_t[i][j] = array_create(layerSizes[i - 1]);
	                for (var k = 0; k < layerSizes[i - 1]; k++) {
	                    m_t[i][j][k] = 0;
	                    v_t[i][j][k] = 0;
	                }
	            }
	        }
    

	    // Увеличиваем шаг времени
	    timestep++;

	    // Обновление весов и смещений
	    for (var i = 1; i < layerCount; i++) {
	        for (var j = 0; j < layerSizes[i]; j++) {
	            for (var k = 0; k < layerSizes[i - 1]; k++) {

	                // Обновляем моменты первого порядка
	                m_t[i][j][k] = beta1 * m_t[i][j][k] + (1 - beta1) * delta[i][j] * output[i - 1][k];
	                // Обновляем моменты второго порядка
	                v_t[i][j][k] = beta2 * v_t[i][j][k] + (1 - beta2) * sqr(delta[i][j] * output[i - 1][k]);

	                // Корректируем моменты
	                var m_t_hat = m_t[i][j][k] / (1 - power(beta1, timestep));
	                var v_t_hat = v_t[i][j][k] / (1 - power(beta2, timestep));

	                // Обновляем веса
	                weights[i][j][k] -= learningRate * m_t_hat / (sqrt(v_t_hat) + epsilon);
	            }

	            // Обновляем смещения
	            var bias_update = bias[i] || 0;
	            // Обновляем моменты для смещений
	            var m_bias = beta1 * (bias_update) + (1 - beta1) * delta[i][j];
	            var v_bias = beta2 * (bias_update) + (1 - beta2) * sqr(delta[i][j]);

	            // Корректируем моменты для смещений
	            var m_t_hat_bias = m_bias / (1 - power(beta1, timestep));
	            var v_t_hat_bias = v_bias / (1 - power(beta2, timestep));

	            // Обновляем смещения
	            bias[i] -= learningRate * m_t_hat_bias / (sqrt(v_t_hat_bias) + epsilon);
			
	        }
	    }
	}
		
#region Loss function
	/// @func MeanSED
	/// @desc Производная функции потерь MSE для массивов
	/// @param {array} predicted - массив предсказанных значений
	/// @param {array} target - массив целевых значений
	/// @return {array} - производная ошибки
	static MeanSED = function(predicted, target) {

	    var derivative = array_create(array_length(predicted));
	    for (var i = 0; i < array_length(target); i++) {
	        derivative[i] = predicted[i] - target[i];
	    }

				//if is_nan(derivative[0])
			//show_message("NAN los function")
	    return derivative;
	}

	/// @func CategoricalCED - CategoricalCrossEntropyDerivative
	/// @desc Производная функции потерь категориальной кросс-энтропии
	/// @param {array} predicted - предсказанные вероятности для каждого класса
	/// @param {array} target - one-hot целевая метка
	/// @return {array} - производная ошибки
	static CategoricalCED = function(predicted, target) {
	    var derivative = array_create(array_length(predicted));

		predicted=Softmax(predicted)
		if mouse_check_button(mb_right)
		show_message(predicted)

	    for (var i = 0; i < array_length(target); i++) {
	        // Убедитесь, что не происходит деления на 0
	        var pred = clamp(predicted[i], .001, 1 - .001);
	        derivative[i] = pred - target[i];
	    }
	    return derivative;
	}
	
/// @func PPOLossD (Derivative)
/// @desc Производная функции потерь PPO
/// @param {array} predicted - текущие предсказанные значения
/// @param {array} target - целевые значения действий
/// @param {array} old_log_probs - старые логарифмы вероятностей
/// @param {array} advantages - преимущества, вычисленные с использованием GAE
/// @param {real} clip_ratio - коэффициент для ограничения изменений
/// @return {array} - массив производных ошибки
static PPOLossD = function(predicted, target, old_log_probs, advantages, clip_ratio) {
    var new_log_probs = array_create(array_length(predicted), 0);
    var derivatives = array_create(array_length(predicted), array_create(array_length(predicted[0]), 0));

    for (var i = 0; i < array_length(predicted); i++) {
        // Используем target для получения вероятности выбранного действия
        var action_prob = predicted[i][target[i]];
        new_log_probs[i] = ln(action_prob);

        var ratio = exp(new_log_probs[i] - old_log_probs[i]);
        var surr1 = ratio * advantages[i];
        var surr2 = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages[i];

        // Loss derivative computation
        var loss_gradient = -min(surr1, surr2); // Negative for gradient descent

        // Обновляем производную для всех действий
        for (var j = 0; j < array_length(predicted[i]); j++) {
            if (j == target[i]) {
                derivatives[i][j] = loss_gradient;
            } else {
                derivatives[i][j] = 0;
            }
        }
    }
    return derivatives;
}
#endregion

}




//// Функция награды
////чем дальше, тем больше в минус уходит награда
//function rewardFunction(pos_agent, pos_goal) {
//    var reward = abs(pos_agent - pos_goal)*0.02;
//    return reward;
//};

// Функция награды
// Агент получает больше награды, когда находится ближе к цели, и меньше награды, когда далеко.
function rewardFunction(pos_agent, pos_goal) {
    var distance = abs(pos_agent - pos_goal);
    var max_distance = 500; // предположим, что максимальное расстояние равно ширине комнаты
    var reward = (max_distance - distance) / max_distance;
    return reward;
};

// Шаг в окружении
function environmentStep(action) {
	var old_state = x/room_width;
    // Обновляем позицию агента на основе действия
    if (action == 0) {
        x -= room_width*0.02; // Движение влево
    } else if (action == 1) {
        x += room_width*0.02; // Движение вправо
    }

    // Получаем награду
    var reward = rewardFunction(x, obj_target.x);

    // Создаём новое состояние
    var next_state = x/room_width;
    
    // Определяем, достигнута ли цель (окончание эпизода)
    var done = abs(x/room_width - obj_target.x/room_width) < 0.1;
    
    return [next_state, reward, done, old_state];
}

// Начальное состояние
position_agent_start = x;

/// @desc Перезапуск окружения и агента
function Restart() {
    // Обнуляем награды и временные шаги
    total_reward = 0;
    timestep = 0;

    // Начальное состояние
    x = position_agent_start;

    // Очистка памяти
    memory.clear();
	
	//state
	current_state=x
}



/// @func StoreData
/// @desc Сохраняет данные состояния, действия, награды, флага завершения, оценки функции ценности и логарифма вероятности
/// @param {array} state - текущее состояние
/// @param {real} action - текущее действие
/// @param {real} reward - награда за действие
/// @param {bool} done - флаг завершения эпизода
/// @param {real} value - оценка функции ценности текущего состояния
/// @param {real} log_prob - логарифм вероятности текущего действия
/// @return {void}
function StoreData(state, action, reward, done, value, log_prob) {
    // Добавляем данные в память
    memory.add(state, action, reward, done, value, log_prob);
}

/// @func SampleAction
/// @desc Выбирает действие на основе вероятностей
/// @param {array} probabilities - массив вероятностей действий
/// @return {int} - выбранное действие
function SampleAction(probabilities) {
    var cumulative_sum = 0;
    var random_value = random(1); // Случайное число от 0 до 1
    var action = 0;

    for (var i = 0; i < array_length(probabilities); i++) {
        cumulative_sum += probabilities[i];
        if (random_value < cumulative_sum) {
            action = i;
            break;
        }
    }
    return action;
}


// Создаём сеть политики
policy_network = new mlp_array([1,3,2],[Tanh,Tanh]);

// Создаём сеть критика (Value network)
value_network = new mlp_array([1,2,1],[Tanh,Tanh]);

// Инициализируем память для PPO
memory = new Memory();

// Функция перезапуска
Restart();
