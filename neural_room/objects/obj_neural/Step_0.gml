/// @description  coment
mlpa.Cost(costt)
mlpa.Forward([1])//на вход mlp подаём данные
//show_message(mlpa.output)	
mlpa.Backward()		
mlpa.Apply(0.1)





// SoftMax
//функция активации
//https://www.youtube.com/watch?v=KpKog-L9veg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=10
/* код из другово проэкта
		switch(global.outputActivation) {
			case OUTPUTA.softMax:
				for(var i = 0; i < OUTPUT; i++) {
					output[i] = softMax(outputRaw[i]);
				}
				break;
			case OUTPUTA.unstableSoftMax:
				for(var i = 0; i < OUTPUT; i++) {
					output[i] = unstableSoftMax(outputRaw[i]);
				}
				break;
		}
		*/