/////////////////////-----------PPO

// Входные данные и целевые значения
//var inputData = [random_range(-1,1), random_range(-1,1)];
//var targetData = [0,1,0]//[0.7, -0.7, 0.4];
policy_network.timestep=0//служит для разгона оптимизаторов (как бы сбрасываю счётчик)
value_network.timestep=0//служит для разгона оптимизаторов (как бы сбрасываю счётчик)


// Шаг PPO
// Инкремент таймстепа
timestep += 1;

// Сохраняем старое состояние
var old_state = x / room_width;

// Выбор действия
policy_network.Forward([old_state]);
var logits = policy_network.output[policy_network.layerCount - 1];
var action_probabilities = Softmax(logits);
var action = SampleAction(action_probabilities);

// Выполняем шаг в окружении
var result = environmentStep(action);
var next_state = result[0];
var reward = result[1];
var done = result[2];

// Обновляем текущее состояние
x = next_state * room_width;

// Вычисление логарифма вероятности действия (log probability)
policy_network.Forward([old_state]);
var action_probs = Softmax(policy_network.output[policy_network.layerCount - 1]);
var log_prob = ln(action_probs[action]);

// Вычисление ценности состояния (Value estimation)
//var val_input=[action,old_state]//если input более 2 значений
value_network.Forward([old_state]);//value_network.Forward([old_state]);//если 1 state
var value = value_network.output[value_network.layerCount - 1][0];

// Сохраняем данные в память
StoreData(old_state, action, reward, done, value, log_prob);

total_reward += reward;



// Обучение PPO
if (timestep % max_timesteps == 0) {
    // Вычисляем возвраты (returns) и преимущества (advantages)
    memory.returns = calculate_returns(memory.rewards, gamma, false, false);//true,false
    memory.advantages = calculate_advantage(memory.rewards, memory.values, memory.returns, gamma, lambda, false);
    show_debug_message($"{memory.advantages}  -advantage");
	show_debug_message($"{memory.rewards}  -rewards");
	//show_debug_message($"{memory.actions}  -actions");
	show_debug_message($"{memory.returns}  -returns")
	
	
	// Создаем минибатчи
    var memory_tuples = memory.tuple();
    for (var epoch = 0; epoch < epochs; epoch++) {
        var mini_batch = memory_tuples//memory.minibatch(memory_tuples, batch_size);

        for (var i = 0; i < array_length(mini_batch); i++) {
            var tuple = mini_batch[i];
            var state = tuple[0];
            var action = tuple[1];
            var reward = tuple[2];
            var done = tuple[3];
            var value = tuple[4];
            var old_log_prob = tuple[5];
            var advantage = tuple[6];
            var return_val = tuple[7];


            // Обновляем политики
			//из-за того что этот код в цикле мы обновляем веса каждый раз, и из-за этого обновляется сеть
			//поэтому new_state ненужен
            policy_network.Forward([state]);
            var new_logits = policy_network.output[policy_network.layerCount - 1];
            var new_action_probabilities = Softmax(new_logits);
            //var new_log_prob = ln(new_action_probabilities[action]);

// //Example
//var predicted = [
//    [0.5, 0.5]

//];
//var target = [1]; // Целевые значения действий
//var old_log_probs = [-0.3567];
//var advantages = [0.9];
//var clip_ratio = 0.2;
//var policy_loss_gradient = policy_network.PPOLossD(predicted, target, old_log_probs, advantages, clip_ratio);

			// Вычисляем производную функции потерь для политики
            var policy_loss_gradient = policy_network.PPOLossD([new_action_probabilities], [action], [old_log_prob], [advantage], clip_ratio);
			

            // Обратное распространение и обновление весов политики
            policy_network.LossFunction(policy_loss_gradient[0]);
            policy_network.Backpropagate(policy_network.AdamUpdateWeights, policy_learning_rate);

            // Обновляем критика
		    //var val_input=[action,state]//если input более 2 значений
            value_network.Forward([state]);//[state]
            var value_loss_gradient = value_network.MeanSED(value_network.output[value_network.layerCount - 1], [return_val]);//!!!!!(value_network.output[value_network.layerCount - 1], [return_val]

            // Обратное распространение и обновление весов критика
            value_network.LossFunction(value_loss_gradient);
            value_network.Backpropagate(value_network.AdamUpdateWeights, value_learning_rate);
        }
    }
    // Перезапуск окружения и агента
    Restart();
}