/// @description Insert description here
// You can write your code in this editor

//карта наград
for (var w=0;w<3;w++)
	for (var h=0;h<3;h++)
	{
		draw_rectangle(800+60*w,200+60*h,800+50+60*w,200+50+60*h,true)
		draw_text(800+60*w+20,200+60*h+20,reward_map[h][w])
		draw_text(800+60*w+20,200+60*h,debugg2[h][w])
	}








