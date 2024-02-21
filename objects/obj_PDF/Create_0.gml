/// @description  гаусовый рандом


array_graf_pfd=[]
for (var i=0;i<60;i++)
{
var bath=[]
var r = gauss(0,1)
//var r =random_normal_distribution() можно и так
var p = pdf(r,0,1)

array_push(bath,r,p)
array_graf_pfd[@ i]=bath
}
//show_message(array_graf_pfd)
//show_message(r)
//show_message(pdf(r,0,1))


//var r=[0.1457, 0.2831, 0.1569, 0.2221, 0.1922]
//var r = 1
//var p = pdf(r,0,1)
//show_message(ln(0.1457))
//show_message(p)