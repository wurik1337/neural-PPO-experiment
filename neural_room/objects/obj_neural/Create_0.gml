/// @description  coment


var layers = [1,2,2,2];
mlpa = new mlp_array(layers);








costt=[0.3,0.1]///правельный ответ
mlpa.Cost(costt)//задаёт правельный ответ, и создаёт переменную delta- это разница между имеется выводом и правельным (в последнем слое)
mlpa.Forward([1])//на вход mlp подаём данные
mlpa.Backward()	//обратная распрост. ошибки - с последнего по первый чёто делает
mlpa.Apply(0.1)//стоханрически градиентный спуск
					
// Batch
batchSize = 25;
batchPosition = 0;