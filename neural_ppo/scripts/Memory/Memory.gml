// Script assets have changed for v2.3.0 see
// https://help.yoyogames.com/hc/en-us/articles/360005277377 for more information
function Memory() constructor {
    // Создание кортежа данных (буфера)
    enum typePPO {
        states,         
        next_states,
        actions,        
        rewards,             
        values,        
        next_values,
        log_probs,
        next_log_probs,
        advantages,     
        returns         
    }
    
    // Буфер для хранения всех параметров
    states = [];
    next_states = [];
    actions = [];
    rewards = [];
    values = [];
    next_values = [];
    log_probs = [];
    next_log_probs = [];
    advantages = [];
    returns = [];
    
    len_memory = 0; // Длина памяти

    // Добавление элементов в память
    static add = function(state, next_state, action, reward, value, next_value, log_prob, next_log_prob) {
        array_push(states, state);
        array_push(next_states, next_state);
        array_push(actions, action);
        array_push(rewards, reward);
        array_push(values, value);
        array_push(next_values, next_value);
        array_push(log_probs, log_prob);
        array_push(next_log_probs, next_log_prob);
        len_memory += 1;
    }

    // Очистка памяти
    static clear = function() {
        array_clear(states);
        array_clear(next_states);
        array_clear(actions);
        array_clear(rewards);
        array_clear(values);
        array_clear(next_values);
        array_clear(log_probs);
        array_clear(next_log_probs);
        array_clear(advantages);
        array_clear(returns);
        len_memory = 0;
    }

    // Удобный вид кортежа данных
    static tuple = function() {
        var array = [];
        for (var i = 0; i < array_length(states); i++) {
            var tupl = [
                states[i],
                next_states[i],
                actions[i],
                rewards[i],
                values[i],
                next_values[i],
                log_probs[i],
                next_log_probs[i],
                advantages[i],
                returns[i]
            ];
            array_push(array, tupl);
        }
        return array;
    }

    // Создание минибатча данных
    static minibatch = function(tuple_array, numb) {
        var array = [];
        for (var i = 0; i < numb; i++) {
            var random_index = irandom_range(0, array_length(tuple_array) - 1);
            array_push(array, tuple_array[random_index]);
        }
        return array;
    }
}


// Вспомогательная функция для вычисления среднего значения массива
function array_mean(array) {
	var sum = 0;
	for (var i = 0; i < array_length(array); i++) {
	    sum += array[i];
	}
	return sum / array_length(array);
}

// Вспомогательная функция для вычисления стандартного отклонения массива
function array_std(array) {
	var _mean = array_mean(array);
	var sum = 0;
	for (var i = 0; i < array_length(array); i++) {
	    sum += sqr(array[i] - _mean);
	}
	return sqrt(sum / array_length(array));
}


// Вспомогательная функция для расчета дисконтированных наград
function calculate_returns(rewards, gamma, normalize, use_discount) {
    var discounted_rewards = array_create(array_length(rewards));
    var running_add = 0;

    if (use_discount) {
        for (var t = array_length(rewards) - 1; t >= 0; t--) {
            running_add = rewards[t] + gamma * running_add;
            discounted_rewards[t] = running_add;
        }
    } else {
        for (var t = 0; t < array_length(rewards); t++) {
            discounted_rewards[t] = rewards[t];
        }
    }

    if (normalize) {
        var _mean = array_mean(discounted_rewards);
        var std = array_std(discounted_rewards);
        for (var t = 0; t < array_length(discounted_rewards); t++) {
            discounted_rewards[t] = (discounted_rewards[t] - _mean) / std;
        }
    }

    return discounted_rewards;
}

// Функция для нахождения минимального значения в массиве
function array_min(arr) {
    var min_val = arr[0];
    for (var i = 1; i < array_length(arr); i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Функция для нахождения максимального значения в массиве
function array_max(arr) {
    var max_val = arr[0];
    for (var i = 1; i < array_length(arr); i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}


/// @func calculate_advantage
/// @desc Вычисляет обобщенные оценки преимущества (GAE) и нормализует их при необходимости
/// @param {array} rewards - массив вознаграждений
/// @param {array} values - массив оценок ценностей состояний
/// @param {array} next_values - массив оценок ценностей следующих состояний
/// @param {real} gamma - коэффициент дисконтирования
/// @param {real} lambda - параметр для обобщенной оценки преимущества
/// @param {bool} use_gae - использовать ли GAE
/// @param {bool} normalize - нормализовать ли значения преимущества
/// @return {array} - массив оценок преимущества
function calculate_advantage(rewards, values, next_values, gamma, lambda, use_gae, normalize) {
    var advantage = array_create(array_length(rewards), 0);

    if (use_gae) {
        var _gae = 0;
        for (var t = array_length(rewards) - 1; t >= 0; t--) {
            var delta = rewards[t] + gamma * next_values[t] - values[t];
            _gae = delta + gamma * lambda * _gae;
            advantage[t] = _gae;
        }
    } else {
        for (var t = 0; t < array_length(rewards); t++) {
            advantage[t] = rewards[t] - values[t]; 
			//advantage[t] = next_values[t] - values[t];
			//default advantage[t] = rewards[t] + gamma * next_values[t] - values[t];

		}
    }

    if (normalize) {
        var _mean = array_mean(advantage);
        var std = array_std(advantage);
        for (var t = 0; t < array_length(advantage); t++) {
            advantage[t] = (advantage[t] - _mean) / std;
        }
    }

    return advantage;
}