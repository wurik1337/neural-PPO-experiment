/// @description  coment
var offset=50 
draw_self()
draw_set_color(c_white)
for (var i=0;i<mlpa.layerCount;i++)
for (var n=0;n<mlpa.layerSizes[i];n++)
draw_circle(x+25*i,y+15*n+offset,7,true)


for (var i=1;i<mlpa.layerCount;i++)
for (var n=0;n<mlpa.layerSizes[i];n++)
for (var s=0;s<mlpa.layerSizes[i-1];s++)
{
var color=mlpa.weights[i][n][s]
var color_rgb=make_color_rgb(0,+color*255,0)
draw_line_color(x+25*i,y+15*n+offset,x+25*(i-1),y+15*s+offset,color_rgb,color_rgb)
}

draw_text(0,room_height/2+offset,string(mlpa.weights)+"-weight")
draw_text(0,room_height/2+offset+40,string(mlpa.bias)+"-bias")
//input
draw_text(x-110,y+offset,"input "+string(mlpa.output[0]))
//output
draw_text(x+100,y+offset,"output "+string(mlpa.output[array_length(mlpa.output)-1]))
//cost
draw_text(x+100,y-40+offset,"cost "+string(examples[index].output))

draw_text(x+100,y-80+offset,"session - on sbrasivaetsa v Apply "+string(mlpa.session))

draw_text(x+100,y-120+offset,"batchError= "+string(batchError*100)+" sumError= "+string(sumError)+" error= "+string(error))
