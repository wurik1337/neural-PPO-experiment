var offset=50 
draw_self()
draw_set_color(c_white)
for (var i=0;i<mlpaA.layerCount;i++)
for (var n=0;n<mlpaA.layerSizes[i];n++)
draw_circle(x+25*i,y+15*n+offset,7,true)


for (var i=1;i<mlpaA.layerCount;i++)
for (var n=0;n<mlpaA.layerSizes[i];n++)
for (var s=0;s<mlpaA.layerSizes[i-1];s++)
{
var color=mlpaA.weights[i][n][s]
var color_rgb=make_color_rgb(0,+color*255,0)
draw_line_color(x+25*i,y+15*n+offset,x+25*(i-1),y+15*s+offset,color_rgb,color_rgb)
}


draw_text(0,room_height/2+offset,string(mlpaA.weights)+"-weight")
draw_text(0,room_height/2+40+offset,string(mlpaA.bias)+"-bias")
//input
draw_text(x-110,y+offset,"input "+string(mlpaA.output[0]))
draw_text(x-110,y+50+offset,"critic input "+string(mlpaC.output[0]))
//output
draw_text(x+100,y+offset+20,"actor output "+string(softmax(actor_out))+"  action "+string(action))//draw_text(x+100,y+offset,"output "+string(softmax(mlpaA.Output()))+"  action "+string(action))
draw_text(x+100,y+offset,"actor output "+string(actor_out))//draw_text(x+100,y+offset,"output "+string(softmax(mlpaA.Output()))+"  action "+string(action))
draw_text(x+100,y+offset+40,"critic output "+string(mlpaC.Output()))
//cost
draw_text(x+100,y-40+offset,"actor cost "+string(mlpaA.delta))//string(mlpaA.delta[array_length(mlpaA.delta)-1]

draw_text(x+100,y-80+offset,"session - Apply "+string(batchPosition))

draw_text(0,room_height/2+70+offset,string(mlpaC.weights)+"-weight critic")
draw_text(0,room_height/2+90+offset,string(mlpaC.bias)+"-bias critic")

draw_text(x,y-50,"reward "+string(reward))


draw_line(x,y,x+lengthdir_x(300,direction),y+lengthdir_y(300,direction))
//batch
//draw_text
draw_set_halign(fa_left)
draw_text_ext(40,40,batch,20,750)
draw_text_ext(40,120,memor,20,750)
draw_set_halign(fa_center)


draw_text(mouse_x+20,mouse_y,position)
draw_text(mouse_x+20,mouse_y+40,action)

//reward map
for (var w=0;w<3;w++)
	for (var h=0;h<3;h++)
	{
		draw_rectangle(400+60*w,200+60*h,400+50+60*w,200+50+60*h,true)
		draw_text(400+60*w+20,200+60*h+20,reward_map[w][h])
		draw_text(400+60*w+20,200+60*h,debugg[w][h])
	}
	

	

draw_sprite(sprite_index,0,400+50+60*(position mod 3)-20,200+50+60*(position div 3)-20)
