randomize()
room_speed=2
memor= new memoryPPO()//картеж данных


function train(memory_array)
{
	//var values=memory_array.minibatch(memory_array.tuple(gamma,true,false),3)
	var values=memory_array.tuple(gamma,false,false)//gamma,discount_reward,normalize
	var val_len = array_length(values)//-1
	var q_vals=array_create(val_len,1)
	var advantage=array_create(val_len,1)
	var actor_loss=array_create(val_len,1)
	var values_old=[]
	var next_values=[]
	var one_hot=[]
	var next_log_prob=[]



	for (var i=0;i<val_len;i++)
	{
		q_vals[i] = values[i][typePPO.reward] + gamma*values[i][typePPO.next_values]//*(1-done)//q_val = memory_array.reward[i] + gamma*q_val*(1.0-done)
	}

	for (var i=0;i<array_length(q_vals);i++)
	{
	values_old[i] = values[i][typePPO.values]
	//next_values[i] = values[i][typePPO.next_values]
	actor_loss[i] =values[i][typePPO.log_prob]
	one_hot[i] = values[i][typePPO.done]
	next_log_prob[i] = values[i][typePPO.next_log_prob]
	aaction[i]= values[i][typePPO.action]
	}

	advantage=calculate_advantages(q_vals,values_old,false)
	//advantage=gae(advantage,0.99,0.95)//улучшение advantage со временем (чем раньше , тем болшьше награды)


	/////критик
	var critic_loss
	critic_loss=array_mean(advantage)//среднее значение массива sqr(array_mean(advantage))


	mlpaC.Forward(aaction[0])
	var debag_c = mlpaC.Cost([critic_loss])//critic_loss
	mlpaC.Backward()



	mlpaA.Forward(aaction[0])


//////////////loss function
var debag_a=mlpaA.Cost_CategorialCE(actor_loss,one_hot,advantage)
//var debag_a=mlpaA.Cost(aactor_loss[0])
//var debag_a=mlpaA.CrossEntropy(actor_loss,one_hot,advantage)
//var debag_a=mlpaA.ppo_loss(next_log_prob,actor_loss,advantage,one_hot,epsilon)


	mlpaA.Backward()

	
	//if batchcount>=batchmax
	//{
	mlpaC.Apply(0.01,true)
	mlpaA.Apply(0.001,true);
	//}




	#region ОТЛАДКА
	batchteg={
	advantag: advantage,
	reward: q_vals,
	critic_los: debag_c,
	actor_los: debag_a,
	log_prb: log_prob,
	value: other.values,
	next_values: other.next_values
	
	}  

	batch=[batchteg]
	#endregion
	
}

fast=false
done=false



//reward
reward=0
epi_reward=0
gamma=0.99
lamda = 0.90
epsilon=0.2

actor_loss=0
log_prob=0
loss_value=0
learn_rate=0.001



entrop=0.2
beta=0.001
alpha=0.001
entropy_coef=0.1

// Batch 
batchSize =1
batchPosition = 0;
batch=[]
batchcount=0
batchmax=5

state=[]
action=0



//сеть актёра
var layers = [2,12,10,9,4];
mlpaA = new mlp_array(layers);

	
//сеть критика
var layers = [mlpaA.layerSizes[0],12,10,12,1];//var layers = [mlpaA.layerSizes[0]+mlpaA.layerSizes[array_length(mlpaA.layerSizes)-1],4,4,1];//в последнем слое ДОЛЖЕН БЫТЬ 1 НЕЙРОН, так как функция Q оценивает ожидаемую долгосрочную награду для конкретного состояния и действия  
mlpaC = new mlp_array(layers);	


function restart()
{
	position=irandom(8)//6
	epi_reward=0
	speed=0
	direction=0
	batchPosition = 0;	// Начинается новая партия.
	batchcount=0 //цикл тренировки с начала
	advantage=0
	log_prob=0
	done=false
	action=""
}



values=0
next_values=0
values_pred=0

dist_reward=330

//тоже отладка
orient=1
noloop=0
adv_memor=[]
adv_memor_pred=0


g=scr_lineGraphCreate(5, 500, 384, 126, "reward_episod", 6, "", "");
g2=scr_lineGraphCreate(5, 600, 384, 126, "Critic", 6, "", "");




position=4//положение комнаты

room_map = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8]
];

reward_map = [
  [1, 0.5, -1],
  [0.5, -1, -1],
  [-1, -1, -1]
];

debugg=[
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]
];



