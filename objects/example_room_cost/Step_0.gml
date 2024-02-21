/// @description Insert description here
// You can write your code in this editor
for(var xx=0;xx<3;xx++)
for (var yy=0;yy<3;yy++)
{	
	//show_message(i)
	//show_message([reward_map[i div 3][i mod 3]])
	Digit.Forward([xx,yy])//на вход mlp подаём данные
	var debag_c = Digit.Cost([reward_map[xx][yy]])//critic_loss
	Digit.Backward()
	Digit.Apply(0.01,true)
	debugg2[xx][yy]=Digit.Output()[0]
}









