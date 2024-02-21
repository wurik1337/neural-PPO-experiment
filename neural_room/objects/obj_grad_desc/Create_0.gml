/// @description  градиентный спуск собственной персоны
//https://realpython.com/gradient-descent-algorithm-python/#basic-gradient-descent-algorithm
//https://www.youtube.com/watch?v=mRjuV3WmMRQ&t=1s
//https://www.youtube.com/watch?v=mRjuV3WmMRQ&list=PLMDIx4RI54PaQPcGIXbhELYmPVW-oVBxy&index=4

//градиентный спуск нужен для минимизации функции потерь. 
//Суть алгоритма – процесс получения наименьшего значения ошибки. 
//Аналогично это можно рассматривать как спуск во впадину в попытке найти золото на дне ущелья (самое низкое значение ошибки).

//градиентный спуск для нахождение минимума
function gradient_descent(start, learn_rate, n_iter)
{
	static gradient = function(p)//функция (математическая а не програмирамная)
	{
		return p*2
	}
	
	var vector=start;
    for (var i=0;i<n_iter;i++)
	{
        diff = -learn_rate * gradient(vector)//(vector*2)//  пример и найдете минимум функции 𝐶 = 𝑣2  это (vector*2).
        vector += diff
	}
    return vector
}



// Сложный Но рабочий пример (градиент замечает яму на графике) функция 4 * power(x,3) - 10 * x - 3
n_iter2=50
start2=0
//чтобы продемострировать
//почему прямая линия , да потому что я рисую круги, и решил ебануть x и y одинаковыми у них
function gradient_descent_draw2(start, learn_rate, n_iter)
{
	static gradient = function(p)//микрофункция , просто которая умножает на 2
	{
		return 4 * power(p,3) - 10 * p - 3//вот это сложная функция //p*2 - простая функция
	}
	var vector=start;
    for (var i=0;i<n_iter;i++)
	{
        diff = -learn_rate * gradient(vector)//(vector*2)//  пример и найдете минимум функции 𝐶 = 𝑣2  это (vector*2).
        vector += diff
		if vector>0
		{
			draw_set_color(c_white)
			draw_circle(x+300-vector*30+160,y+100-gradient(vector),2,true) //умножение регулируй это чтобы лучше видеть график
		}
		else
		{
			draw_set_color(c_red)
			draw_circle(x+300-vector*30+160,y+100+gradient(vector),2,true)	//умножение регулируй это чтобы лучше видеть график
		}
	}
    return vector
}






////////////////простая функция x*2
n_iter=50
learn_rate=0.1
start=100
function gradient_descent_draw(start, learn_rate, n_iter)
{
	static gradient = function(p)//микрофункция , просто которая умножает на 2
	{
		return p*2 //- простая функция
	}
	
	var vector=start;
    for (var i=0;i<n_iter;i++)
	{
        diff = -learn_rate * gradient(vector)//(vector*2)//  пример и найдете минимум функции 𝐶 = 𝑣2  это (vector*2).
        vector += diff
		if vector>0
		{
			draw_set_color(c_white)
			draw_circle(x+100-vector+160,y+100-gradient(vector),2,true) 
		}
		else
		{
			draw_set_color(c_red)
			draw_circle(x+100-vector+160,y+100+gradient(vector),2,true)	
		}
		

	}
    return vector
}
show=gradient_descent_draw(start,learn_rate,n_iter)


/////////////////////3 пример на простую нейронную сеть
input=4
weight=0.4
win=12.6//правильный ответ для out
learn_rat=0.1
var error=0//объявил переменную
var gradient=0//объявил переменную
for (var i=0;i<10;i++)
{
	out=input*weight//имитация 1 нейрона (вход*массу)
	delta=out-win//ЧИСЛАЯ ОШИБКА -разница между правильным и выходным слоем , служит как величина для каректировки вход. значений
	weight_delta=delta*input//служит для оценки величины и направления изменения веса, большую или меньшую сторону -44
	weight+=-weight_delta*learn_rat//берём часть от направления и делаем поправку весов туда
	
}
//show_message("error "+string(error) + " weight " + string(weight)+ " out " + string(out))

