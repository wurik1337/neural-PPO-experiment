/// @description  coment
randomise()
if alarm[0]=-1
if (k < 10) {
    //начальное состояние (если вы хотите изменить начальную комнату, измените это значение на начальную комнату.)
    state = 0;


    while (state != 6) { //цикл пока до финал комнаты не дойдёт
      if (state == 0) {
        action = 1;
      }
      else if (state == 1) {
        r = irandom(100) % 3;
        action = M[1][r];
      }
      else if (state == 2) {
        r = irandom(100) % 3;
        action = M[2][r];
      }
      else if (state == 3) {
        r = irandom(100) % 2;
        action = M[3][r];
      }
      else if (state == 4) {
        r = irandom(100) % 3;
        action = M[4][r];
      }
      else if (state == 5) {
        action = 4 ;
      }
      else if (state == 6) {
        action = 2 ;
      }
      a = -10;
      for (i = 0; i < array_length(Q); i++) { //я хз,ищем наибольшее число в цикле
        if (a < Q[action][i]) {//пробегает по масивам в ШИРИНУ В зависимости от действия
          a = Q[action][i];  //и сохраняем
        }
      }

      Qmax = a * gammma;
      Q[state][action] = R[state][action] + Qmax;
      state = action;

	  //отладка
	  alarm[0]=40
	  show_debug_message(state)
	  array_push(state_view,state)//чтобы посмотреть
	  count++
	  //var r=get_string(array_length(Q),"awd")
	  if count div 3
	  {
	  show_debug_message("---")
	  count = 0
	  }
	  //-
    }
    k++;
}
else
{ 

    //начальное состояние (если вы хотите изменить начальную комнату, измените это значение на начальную комнату.)
    state = 0;
    //Находит действие с наибольшим количеством очков и выполняет это действие
    b = -10;
    while (state != 6) { //цикл нужен для того чтобы не возравщатся в начало step, иначе будет повторятся заново условие выше
      for (i = 0; i < array_length(Q); i++) {
        if (b <= Q[state][i]) {
          b = Q[state][i];//пробегает по масивам в ШИРИНУ В зависимости от комнаты
          l = i;//сохраняется наибольшее число в ширину(это будет правильная комната)

		  alarm[0]=40//для медлености
        }
      }
      state = l;//сохраняет прошлый результат

	  //отладка
	  show_debug_message(state)
	  array_push(state_view,state)//чтобы посмотреть
	  count++
	  if count div 3
	  {
	  show_debug_message("---")
	  count = 0
	  }
	  //-
    }
}



