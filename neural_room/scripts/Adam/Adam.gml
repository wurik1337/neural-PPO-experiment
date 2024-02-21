// Ресурсы скриптов были изменены для версии 2.3.0, подробности см. по адресу
// https://help.yoyogames.com/hc/en-us/articles/360005277377
function Adam(){
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	

	/// @func	Apply(learnRate);
	/// @desc	Применяет средние градиенты к весам MLP.
	Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient, unbiasOne, unbiasTwo;
		var epsilon = .001;	// Чтобы не делить на 0.
		var divider = 1 / trainingSession;						// Кэширование.
		var oneDivider = 1 / (1 - power(betaOne, iteration));	// 
		var twoDivider = 1 / (1 - power(betaTwo, iteration));	// 
		
		// Обновите веса и смещения.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			// Рассчитать импульс (возможно как в Momentum)
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;
			oneMoment[@i][@j][@k] = betaOne * oneMoment[i][j][k] + (1 - betaOne) * gradient;
			twoMoment[@i][@j][@k] = betaTwo * twoMoment[i][j][k] + (1 - betaTwo) * gradient * gradient;
			// Коррекция смещения
			unbiasOne = oneMoment[i][j][k] * oneDivider;
			unbiasTwo = twoMoment[i][j][k] * twoDivider;
			// AdaGrad.
			weights[@i][@j][@k] += -learnRate * unbiasOne / (sqrt(unbiasTwo) + epsilon);
		}}}
				
		trainingSession = 0;	// Счет начинается снова.
		session++;		// Сделано на одну итерацию Адама больше
	}	
}