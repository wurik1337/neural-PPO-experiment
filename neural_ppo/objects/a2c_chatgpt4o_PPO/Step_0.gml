/////////////////////-----------PPO

// Шаг PPO
// Инкремент таймстепа
timestep += 1;

// Сохраняем старое состояние
var old_state=getState()

// Выбор действия
policy_network.Forward(old_state);
var logits = policy_network.output[policy_network.layerCount - 1];
var action_probabilities = Softmax(logits);
var action = SampleAction(action_probabilities);

// Выполняем шаг в окружении
var result = environmentStep(old_state,action);
var next_state = result[0];
var reward = result[1];
var done = result[2];

// Обновляем текущее состояние
x = next_state[0] * room_width;
y = next_state[1] * room_height;

// Вычисление логарифма вероятности действия (log probability)
policy_network.Forward(old_state);
var action_probs = Softmax(policy_network.output[policy_network.layerCount - 1]);
var log_prob = ln(action_probs[action]);

// Вычисление ценности состояния (Value estimation)
value_network.Forward(old_state);
var value = value_network.output[value_network.layerCount - 1][0];

// Сохраняем данные в память
StoreData(old_state, action, reward, done, value, log_prob);

total_reward += reward;



// Обучение PPO
if (timestep % max_timesteps == 0) {
    // Вычисляем возвраты (returns) и преимущества (advantages)
	// GAE и Discount плохо работают, но вот normalize все хорошо работают
    memory.returns = calculate_returns(memory.rewards, gamma, false, false);//true, false
    memory.advantages = calculate_advantage(memory.rewards, memory.values, memory.returns, gamma, lambda, false, false);//true true
    //show_debug_message($"{memory.advantages}  -advantage");
	//show_debug_message($"{memory.rewards}  -rewards");
	//show_debug_message($"{memory.actions}  -actions");
	//show_debug_message($"{memory.returns}  -returns")
	//show_debug_message($"{memory.states}  -stats")
	
	
	// Создаем минибатчи
    var memory_tuples = memory.tuple();
    for (var epoch = 0; epoch < epochs; epoch++) {
        var mini_batch = memory.minibatch(memory_tuples, batch_size);

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
            policy_network.Forward(state);
            var new_logits = policy_network.output[policy_network.layerCount - 1];
            var new_action_probabilities = Softmax(new_logits);
			
			// Вычисляем производную функции потерь для политики
            var policy_loss_gradient = policy_network.PPOLossD([new_action_probabilities], [action], [old_log_prob], [advantage], clip_ratio);
			

            // Обратное распространение и обновление весов политики
            policy_network.LossFunction(policy_loss_gradient[0]);
            policy_network.Backpropagate(policy_network.AdamUpdateWeights, policy_learning_rate);

            // Обновляем критика
            value_network.Forward(state);
            var value_loss_gradient = value_network.MeanSED(value_network.output[value_network.layerCount - 1], [return_val]);
			
            // Обратное распространение и обновление весов критика
            value_network.LossFunction(value_loss_gradient);
            value_network.Backpropagate(value_network.AdamUpdateWeights, value_learning_rate);

        }
    }
    // Перезапуск окружения и агента
    Restart();
}