/// @description  coment
draw_text(x,y,"gradient descent") 
draw_text(x,y+40,"out " + string(show))
draw_text(x,y+20,"learn_rate " + string(learn_rate))
draw_set_color(c_white)
draw_text(x+200,y,"grafik 1")
gradient_descent_draw(start,learn_rate,n_iter)


draw_text(x+350,y,"grafik 2")
gradient_descent_draw2(start2,learn_rate,n_iter2)
