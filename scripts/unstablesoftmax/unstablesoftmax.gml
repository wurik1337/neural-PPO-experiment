// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://help.yoyogames.com/hc/en-us/articles/360005277377
function unstableSoftMax(o){
	
	var bottom = 0;
	for(var i = 0; i < OUTPUT.size; i++) {
		bottom += exp(outputRaw[i]);
	}
	
	return exp(o) / bottom;

}