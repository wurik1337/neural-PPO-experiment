// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://help.yoyogames.com/hc/en-us/articles/360005277377
function array_sum(array) {//сложение
	var result = 0;
	for (var i=0; i<array_length(array); i++) {
		result += array[i];
	}
	return result;
}

function array_sub(array1,array2) {//вычитание subtract но только чтобы 2 масива были одинаковы по дленне
var array=array1
	for (var i=0; i<array_length(array2); i++) {
		array[i] -= array2[i];
	}
	return array;
}

