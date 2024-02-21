/// @description  coment
draw_set_color(c_white)
for (var i=0;i<mlpa.layerCount;i++)
for (var n=0;n<mlpa.layerSizes[i];n++)
draw_circle(x+25*i,y+15*n,7,true)


for (var i=1;i<mlpa.layerCount;i++)
for (var n=0;n<mlpa.layerSizes[i];n++)
for (var s=0;s<mlpa.layerSizes[i-1];s++)
{
var color=mlpa.weights[i][n][s]
var color_rgb=make_color_rgb(0,+color*255,0)
draw_line_color(x+25*i,y+15*n,x+25*(i-1),y+15*s,color_rgb,color_rgb)
}

draw_text(0,room_height/2,string(mlpa.weights)+"-weight")
draw_text(0,room_height/2+40,string(mlpa.bias)+"-bias")
//input
draw_text(x-110,y,"input "+string(mlpa.output[0]))
//output
draw_text(x+100,y,"output "+string(mlpa.output[array_length(mlpa.output)-1]))
//cost
draw_text(x+100,y-40,"cost "+string(costt))

draw_text(x+100,y-80,"session - on sbrasivaetsa v Apply "+string(mlpa.session))

