/// @description  coment
action=[x,y]//действие
batchPosition++;



////////////////////////////////////////////////////batch
// Calculate batch.

	// Get prediction
	mlpa.Forward( examples[index].input );
	// Calculate error delta for output layer
	error=mlpa.Cost( examples[index].output );//error для того чтобы функиця Cost возвращал ошибку
	sumError += error/10;
	// Backpropagate output-error through network
	mlpa.Backward();
	// Choose next example
	index = (index+1) mod array_length(examples);


//BATCH
// Применение градиентов и дельт для обновления весов и смещений.
if (batchPosition >= batchSize) {
	mlpa.Apply(0.1);//стахастический градиентный спуск ,а в скобках скорость обучения
	batchError = sumError / batchPosition
	sumError = 0;
	batchPosition = 0;	// Начинается новая партия.
}// Вам не нужно зацикливаться на одном шаге, но вы можете накапливать партию в течение нескольких шагов. Просто не применяйте до того, как партия будет сделана.
// Также вам не нужно идти по порядку.

