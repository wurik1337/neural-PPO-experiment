// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://help.yoyogames.com/hc/en-us/articles/360005277377
#region генерации случайных значений из нормального распределения.
function gauss(m=0, sd=1)
{
	//https://www.gmlscripts.com/script/gauss
	//https://stackoverflow.com/questions/41974615/how-do-i-calculate-pdf-probability-density-function-in-python
	//m-среднее значение
	//sd - отклонение
    var x1, x2, w;
    do {
        x1 = random(2) - 1;
        x2 = random(2) - 1;
        w = x1 * x1 + x2 * x2;
    } until (0 < w && w < 1);
 
    w = sqrt(-2 * ln(w) / w);
    return m + sd * x1 * w;
}

function random_normal_distribution(m=0, sd=1)///типо Gauss Сгенерировать случайное число из нормального распределения (mean, stddev) 
{//создан нейросетью
    var u1 = random_range(0, 1);
    var u2 = random_range(0, 1);
    var z1 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2);
    return m + sd * z1; // Возвращаем только первое значение, z1
}
#endregion


//тоже самое log_prob!!!нормального распределения normal_distribution
function pdf(val, men=0, stdev=1)//x,mean=0 stdev=1 (standard deviation))
{//ПРИМЕР pdf(OrnsteinUhlenbeckActionNoise(action[0]))
	//также служит оценкой распределения
//служит чтобы показать изменение вероятности на каком-то конкретном элементе множества.
//тут она типа "normal",но есть и другие типы
//(чем ближе к центру mean тем больше значение (так как больше шанс выпада))
	var exponent=0
	exponent = exp(-(sqr(val-men) / (2 * sqr(stdev) )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
}

//РАБОЧИЙ (не удаляй) это PDF!!!
function log_probb(val,mu=0,sigma=1)//!!!!!!вроде пока лучший вариант, так как нашла нейронка
{
	
	//VERSION 1 это я нашёл первой, но не уверен так как не пробовал
return -sqr(val - mu) / (2 * sqr(sigma)) - 0.5 * ln(2 * pi) - ln(sigma)
//было return -sqr(val - mu) / (2 * sqr(sigma)) - 0.5 * ln(2 * pi * sqr(sigma))
}



function OrnsteinUhlenbeckActionNoise(val,mu=0,sigma=1)//РАБОЧИЙ
{//ornstein-uhlenbeck noise
	var theta=.15
	var dt=0.01
	var dx = -theta * (val - mu) * dt + sigma * sqrt(dt) * gauss(mu,sigma);
	return val + dx;
}

function array_standard_deviation(arr)
{
	/// @param arr Массив чисел
	///это скрипт чтобы найти STD (в распределение есть mean и std)
	var m = 0;//mean
	var n = array_length(arr);

	// Вычисляем среднее значение
	for (var i = 0; i < n; i++)
	{
	    m += arr[i]+random(0.001);//немного шума
	}
	m /= n;

	// Вычисляем сумму квадратов разностей от среднего
	var sum_squared_diff = 0;
	for (var i = 0; i < n; i++)
	{
	    sum_squared_diff += sqr(arr[i] - m);
		//show_message(sqr(arr[i] - m))
	}

	// Вычисляем стандартное отклонение
	var standard_deviation = sqrt(sum_squared_diff / n);

	return standard_deviation;
}








