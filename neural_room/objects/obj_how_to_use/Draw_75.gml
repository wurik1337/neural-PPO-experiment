/// @desc b) TRAIN WITH BATCH 
// Таким образом партия рассчитывается за несколько шагов
// Веса и смещения обновляются только после завершения пакета.

// Get prediction
mlpa.Forward( examples[index].input );
mlpg.Forward( examples[index].input );

// Calculate error delta for output layer
mlpa.Cost( examples[index].output );
mlpg.Cost( examples[index].output );

// Backpropagate output-error through network
mlpa.Backward();
mlpg.Backward();
	
// Выберите следующий пример. Предварительная партия.
index = (index+1) mod array_length(examples);
batchPosition++;

//BATCH
// Применение градиентов и дельт для обновления весов и смещений.
if (batchPosition >= batchSize) {
	mlpa.Apply(.1);
	mlpg.Apply(.1);
	batchPosition = 0;	// New batch starts.
}




