/// @description  coment
if keyboard_check_pressed(ord("1"))
{
for(var xx=0;xx<3;xx++)
	for (var yy=0;yy<3;yy++)
	{	
			var inp_crit = [xx,yy]
			mlpaC.Forward(inp_crit)//на вход mlp подаём данные
			debugg[xx][yy]=mlpaC.Output()[0]
	}
}


batchPosition++;


//REWARD
if position>-1 and position<9 
reward=reward_map[position mod 3][position div 3] 
else
reward=-1



#region
state=[position mod 3, position div 3]



var inp_crit = state//array_concat(state,llog_prob)
mlpaC.Forward(inp_crit)
values=mlpaC.Output()[0]

//////////АКТЁР (прогнозирование)
mlpaA.Forward(state)
actor_out=mlpaA.Output()

log_prob=softmax(actor_out)
var llog_prob=argmax(log_prob)

x=700
y=300


#region Move

if llog_prob[0]=1
	if (position mod 3)>0
	{
		position-=1
		action="left"
	}
	else
	{
	position=10
	action="leftgg"
}

if llog_prob[1]=1
	if (position mod 3)!=2
	{
		position+=1
		action="right"
	}
	else
	{
	position=10
	action="rightgg"
}
if llog_prob[2]=1
{
	if (position div 3)>0
	{
	position-=3
	action="up"
	}
	else
	{
	position=10
	action="upgg"	
	}
}
if llog_prob[3]=1
{
	if (position div 3)!=2
	{
	position+=3
	action="down"
	}
	else
	{
	position=10
	action="downgg"	
	}
}
#endregion


epi_reward+=reward


	
scr_lineGraphValueAdd(g2, values);


////////////////////////////////NEXT STATION and NEXT ACTION
next_state=[position mod 3, position div 3]

mlpaA.Forward(next_state)//////////АКТЁР (прогнозирование)
actor_out=mlpaA.Output()
new_log_prob=softmax(actor_out)
var nllog_prob=argmax(new_log_prob)



var inp_crit = next_state//array_concat(next_state,nllog_prob)
mlpaC.Forward(inp_crit)
next_values=mlpaC.Output()[0]


//memory bath
memor.add(next_values,log_prob,new_log_prob,values,reward,state,llog_prob)



if (batchPosition div batchSize) = 1 or done
{
	batchPosition=0
	batchcount+=1

	
	if reward=-1
	instance_create_depth(600,200,-10,obj_reward,{
	color: c_red
	})
	if reward=1
	instance_create_depth(600,200,-10,obj_reward,{
	color: c_green
	})
	
	scr_lineGraphValueAdd(g, epi_reward);

	
	self.train(memor)
    memor.clear()

	if batchcount>=batchmax
	restart()

}
	
#endregion



