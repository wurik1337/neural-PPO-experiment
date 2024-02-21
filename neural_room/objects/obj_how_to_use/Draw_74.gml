/// @desc a) TRAIN WITH BATCH

// (mlp_mini doesn't support batching.)

// Calculate batch.
repeat(batchSize) {
	// Get prediction
	mlpa.Forward( examples[index].input );
	mlpg.Forward( examples[index].input );

	// Calculate error delta for output layer
	mlpa.Cost( examples[index].output );
	mlpg.Cost( examples[index].output );

	// Backpropagate output-error through network
	mlpa.Backward();
	mlpg.Backward();
	
	// Choose next example
	index = (index+1) mod array_length(examples);
}

// После того, как пакет выполнен, применяем кумулятивные значения.
// Применить автоматически вычисляет среднее из них.
// Применяем градиенты и дельты для обновления весов и смещений.
mlpa.Apply(.1);		// Скорость обучения не должна быть слишком большой, чтобы не выйти за пределы нормы.
mlpg.Apply(.1);		// Also too small increases time used for learning.


// Это берет мини-пакет всех примеров, в конце концов перебирает их и начинает заново.
// Вам не нужно зацикливаться на одном шаге, но вы можете накапливать партию в течение нескольких шагов. Просто не применяйте до того, как партия будет сделана.
// Также вам не нужно идти по порядку.

