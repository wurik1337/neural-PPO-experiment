

/// @func	Tanh(input); //КСТАТИ , всё верно!), я проверял по формуле с википедии и всё получилось как сдесь (там имеет другой вид)
/// @desc	Squishes value to be between -1 and +1 as S-curve. Similiar to Sigmoid
function Tanh(input) {
	return ((2 / (1 + exp(-2 * input))) - 1);

}

/// @func	TanhDerivative(input); /// по другому производная
function TanhDerivative(input) {
	return (1 - sqr(Tanh(input)));
}
