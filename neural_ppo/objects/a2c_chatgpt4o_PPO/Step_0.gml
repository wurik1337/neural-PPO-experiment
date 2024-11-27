/////////////////////-----------PPO
// Шаг PPO
// Инкремент таймстепа
timestep += 1;

// Сохраняем старое состояние
var old_state=getState()

// Выбор действия
policy_network.Forward(old_state);
var logits = policy_network.output[policy_network.layerCount - 1];
action_probabilities = Softmax(logits);
var action = SampleAction(action_probabilities);


// Выполняем шаг в окружении
var result = environmentStep(action);
var next_state = result[0];
var reward = result[1];
var done = result[2];


// Вычисление логарифма вероятности действия (log probability)
var log_prob = ln(action_probabilities[action]);

////// Вычисление логарифма вероятности действия (next_log probability)
//logits = policy_network.Forward(next_state);
//action_probs = Softmax(logits);
//var next_log_prob = ln(action_probs[SampleAction(action_probs)]);


// Вычисление ценности состояния (Value estimation)
value_network.Forward(old_state);
var value = value_network.output[value_network.layerCount - 1][0];

// Вычисление будущую ценность состояния (next_Value)
value_network.Forward(next_state);
var next_value = value_network.output[value_network.layerCount - 1][0];


// Сохраняем данные в память
StoreData(old_state, action, reward, next_state, value, next_value,action_probabilities,0);

total_reward += reward;



// Обучение PPO
if (timestep % max_timesteps == 0) {
    // Вычисляем возвраты (returns) и преимущества (advantages)
	// GAE и Discount плохо работают, но вот normalize все хорошо работают
    memory.returns = calculate_returns(memory.rewards, gamma, false, false);
	//memory.returns = log_scale_rewards(memory.returns)
	//memory.returns = normalize_episode_rewards(memory.returns)

    memory.advantages = calculate_advantage(memory.returns, memory.values, memory.next_values, gamma, lambda, false, false);
	//memory.advantages = log_scale_rewards(memory.advantages)
	//memory.advantages = center_advantages(memory.advantages)

	if keyboard_check(vk_alt)
	show_message(["return",memory.returns,"advantage",memory.advantages])



	// Создаем минибатчи
    var memory_tuples = memory.tuple();
    for (var epoch = 0; epoch < epochs; epoch++) {
		var mini_batch = memory.minibatch(memory_tuples, batch_size);

		for (var i = 0; i < array_length(mini_batch); i++) {
		    var tuple = mini_batch[i];
		    var state = tuple[0];
		    var next_state = tuple[1];
		    var action = tuple[2];
		    var reward = tuple[3];
		    var value = tuple[4];
		    var next_value = tuple[5];
		    var log_prob = tuple[6];
		    var next_log_prob = tuple[7];
		    var advantage = tuple[8];
		    var return_val = tuple[9];

	


            // Обновляем политики
			//из-за того что этот код в цикле мы обновляем веса каждый раз, и из-за этого обновляется сеть
			//поэтому new_state ненужен
            policy_network.Forward(next_state);
            var new_logits = policy_network.output[policy_network.layerCount - 1];
            var new_action_probabilities = Softmax(new_logits);
			
			//это для проверки, пусть будет 
			//а вот Forward(state) неубирай, он нужен для обнавления весов
			policy_network.Forward(state);
			var total_prediction = policy_network.output[policy_network.layerCount - 1];
            var total_prediction_softmax = Softmax(total_prediction);
			
			// Вычисляем производную функции потерь для политики
            var policy_loss_gradient = policy_network.PPOLossD([new_action_probabilities], [action], [log_prob], [advantage], clip_ratio,[total_prediction_softmax],total_prediction);//[total_prediction] //logits

			

            // Обратное распространение и обновление весов политики
            policy_network.LossFunction(policy_loss_gradient[0]);
            policy_network.Backpropagate(policy_network.AdamUpdateWeights, policy_learning_rate);

            // Обновляем критика
            value_network.Forward(state);
            var value_loss_gradient = value_network.MeanSED(value_network.output[value_network.layerCount - 1], [return_val]);//return_val//return_val+gamma*next_value

			
            // Обратное распространение и обновление весов критика
            value_network.LossFunction(value_loss_gradient);
            value_network.Backpropagate(value_network.AdamUpdateWeights, value_learning_rate);

        }
    }
    // Перезапуск окружения и агента
    Restart();

}