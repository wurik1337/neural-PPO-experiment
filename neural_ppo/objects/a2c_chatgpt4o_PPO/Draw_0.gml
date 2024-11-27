/// @description  coment
draw_self()

var offset=50 
draw_self()
draw_set_color(c_white)
for (var i=0;i<policy_network.layerCount;i++)
for (var n=0;n<policy_network.layerSizes[i];n++)
draw_circle(x+25*i,y+15*n+offset,7,true)


for (var i=1;i<policy_network.layerCount;i++)
for (var n=0;n<policy_network.layerSizes[i];n++)
for (var s=0;s<policy_network.layerSizes[i-1];s++)
{
var color=policy_network.weights[i][n][s]
var color_rgb=NeuronColor(color)//make_color_rgb(0,+color*255,0)
draw_line_color(x+25*i,y+15*n+offset,x+25*(i-1),y+15*s+offset,color_rgb,color_rgb)
}
draw_text(x,y+20,action_probabilities)//вероятность действий

// Отображение текущей информации
draw_text(10, 10, "Position: " + string(x));
draw_text(10, 30, "Goal: " + string(obj_target.x));
draw_text(10, 50, "Total Reward: " + string(total_reward));
draw_text_ext(10, 70,$"{value_network.output}  -Output value",15,1000)
draw_text_ext(10,180,$"{policy_network.output}  -Output policy",30,1000)

//draw_text_ext(10,300,$"{my_network.output} {tree} -MY Output policy",30,1000)