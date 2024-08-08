/// @description  Функции, гиперпараметры и т.п.
randomize()


function mlp_array(layerSizeArray, activationFunctionsArray) constructor {
    #region /// INITIALIZE NETWORK - Neurons, weights and biases.
    session = 0;
    layerCount = array_length(layerSizeArray); // число слоёв
    layerSizes = array_create(layerCount); // число нейронов в слое
    array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);

    // Initialize neurons + their deltas.
    activity = []; // активации нейронов
    output = [];
    bias = []; // нейроны смещения
    delta = []; // Разница между фактическим выходом и желаемым результатом. Рассчитано из текущего примера
    timestep = 0; // служит для усиления моментов в оптимизаторах (SGD, Adam и т.д.)

    for (var i = 0; i < layerCount; i++) {
        activity[i] = array_create(layerSizes[i]);
        output[i] = array_create(layerSizes[i]);
        delta[i] = array_create(layerSizes[i]);
        bias[i] = array_create(layerSizes[i]);
        for (var j = 0; j < layerSizes[i]; j++) {
            bias[i][j] = random_range(-0.2, 0.2);
        }
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
                weights[i][j][k] = random_range(-0.5, 0.5); // Give default random weight.
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
            activity[i][j] = 0; // Инициализируем сумму активности для нейрона

            for (var k = 0; k < layerSizes[i - 1]; k++) {
                // Суммируем взвешенные входные данные
                activity[i][j] += output[i - 1][k] * weights[i][j][k];
            }
            
            // Применяем соответствующую функцию активации к сумме активностей + смещение
            output[i][j] = activationFunctions[i - 1](activity[i][j] + bias[i][j]);
        }
    }

    // Возвращаем выходной массив последнего слоя
    return output[layerCount - 1];
}

static Output = function() {
    return output[layerCount - 1]; // Возвращаем выходные данные последнего слоя
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
            delta[i][j] = errorSum * activationFunctionsDerivative[i](activity[i][j]);
        }
    }

    // Обновляем веса и смещения
    updateWeights(learningRate);
}


static SGDUpdateWeights = function(learningRate) {
    for (var i = 1; i < layerCount; i++) {
        for (var j = 0; j < layerSizes[i]; j++) {
            for (var k = 0; k < layerSizes[i - 1]; k++) {
                weights[i][j][k] -= learningRate * delta[i][j] * output[i - 1][k];
            }
            bias[i][j] -= learningRate * delta[i][j];
        }
    }
}


// Инициализация параметров для Adam вне функции
 m_t = []; // Моменты первого порядка
 v_t = []; // Моменты второго порядка
 m_t_bias = []; // Моменты первого порядка для смещений
 v_t_bias = []; // Моменты второго порядка для смещений
 epsilon = .001; // Чтобы не делить на 0.
 beta1 = .9; // коэффициент для моментов первого порядка
 beta2 = .999; // коэффициент для моментов второго порядка
 timestep=0;

// Инициализация m_t, v_t, m_t_bias и v_t_bias (вроде можно убрать)
for (var i = 1; i < layerCount; i++) {
    m_t[i] = [];
    v_t[i] = [];
    m_t_bias[i] = array_create(layerSizes[i]);
    v_t_bias[i] = array_create(layerSizes[i]);

    for (var j = 0; j < layerSizes[i]; j++) {
        m_t[i][j] = array_create(layerSizes[i - 1]);
        v_t[i][j] = array_create(layerSizes[i - 1]);
        for (var k = 0; k < layerSizes[i - 1]; k++) {
            m_t[i][j][k] = 0;
            v_t[i][j][k] = 0;
        }
        m_t_bias[i][j] = 0;
        v_t_bias[i][j] = 0;
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

            // Обновляем моменты для смещений
            m_t_bias[i][j] = beta1 * m_t_bias[i][j] + (1 - beta1) * delta[i][j];
            v_t_bias[i][j] = beta2 * v_t_bias[i][j] + (1 - beta2) * sqr(delta[i][j]);

            // Корректируем моменты для смещений
            var m_t_hat_bias = m_t_bias[i][j] / (1 - power(beta1, timestep));
            var v_t_hat_bias = v_t_bias[i][j] / (1 - power(beta2, timestep));

            // Обновляем смещения
            bias[i][j] -= learningRate * m_t_hat_bias / (sqrt(v_t_hat_bias) + epsilon);
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
        new_log_probs[i] = ln(action_prob);//action_prob-1;

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
// Чем ближе агент к цели по осям x и y, тем выше награда
function rewardFunction(pos_agent, pos_goal) {
    var distance_x = abs(pos_agent[0] - pos_goal[0]); // расстояние по оси x
    var distance_y = abs(pos_agent[1] - pos_goal[1]); // расстояние по оси y

    // Общая дистанция как евклидово расстояние
    var distance = sqrt(distance_x * distance_x + distance_y * distance_y);

    // Максимальное расстояние - диагональ прямоугольной области (комнаты)
    var max_distance = sqrt(room_width * room_width + room_height * room_height);

    // Награда: чем ближе агент к цели, тем выше награда (от 0 до 1)
    var reward = (max_distance - distance) / max_distance;
    return reward;
}
	

// Шаг в окружении
function environmentStep(old_state,action) {
    // Обновляем позицию агента на основе действия
    if (action == 0) {
        x -= room_width * 0.02; // Движение влево
    } else if (action == 1) {
        x += room_width * 0.02; // Движение вправо
    } else if (action == 2) {
        y -= room_height * 0.02; // Движение вверх
    } else if (action == 3) {
        y += room_height * 0.02; // Движение вниз
    }

    // Получаем награду
    var reward = rewardFunction([x, y], [obj_target.x, obj_target.y]);

    // Создаём новое состояние
    var next_state = getState();

    // Определяем, достигнута ли цель (окончание эпизода)
    var done = (abs(x / room_width - obj_target.x / room_width) < 0.1) && (abs(y / room_height - obj_target.y / room_height) < 0.1);
    
    return [next_state, reward, done, old_state];
}

function getState() {
	return [x / room_width, y / room_height]; //[obj_target.x / room_width, obj_target.y / room_height,x / room_width, y / room_height];
}

// Начальное состояние
x_position_agent_start = x;
y_position_agent_start = y;

///Перезапуск окружения и агента
function Restart() {
    // Обнуляем награды и временные шаги
    total_reward = 0;
    timestep = 0;

    // Начальное состояние
    x = x_position_agent_start;
	y = y_position_agent_start;
	

    // Очистка памяти
    memory.clear();
	
	policy_network.timestep=0//служит для разгона оптимизаторов 
	value_network.timestep=0//служит для разгона оптимизаторов 

	//with obj_target //чтобы цель были в разных местах
	//{
	//	x=random(room_width)
	//	y=random(room_height)
	//}
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


// Гиперпараметры PPO
gamma = 0.99; // Коэффициент дисконтирования
lambda = 0.95; // Коэффициент для GAE
clip_ratio = 0.2; // Коэффициент для ограничения изменений
policy_learning_rate = 0.01;//0.01
value_learning_rate = 0.1;//0.04
batch_size = 40; //20  Размер минибатча
epochs = 1; // Количество эпох для обновления

// Инициализация дополнительных переменных
training = false;
total_reward = 0;
timestep = 0;
max_timesteps = 100; //40 Количество шагов до обновления

// Создаём сеть политики
policy_network = new mlp_array([2,5,4,4],[Tanh,Tanh,Tanh]);//policy_network = new mlp_array([2,3,4],[Tanh,Tanh]);

// Создаём сеть критика (Value network)
value_network = new mlp_array([2,5,4,1],[Tanh,Tanh,Tanh]);

// Инициализируем память для PPO
memory = new Memory();

// Функция перезапуска
Restart();