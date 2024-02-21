/// @description scr_lineGraphCreate(x, y, w, h, xLabel(string), xIncrement, yLabel(string), title(string));
/// @param x
/// @param y
/// @param w
/// @param h
/// @param xLabel
/// @param xIncrement
/// @param yLabel
/// @param title
function scr_lineGraphCreate(argument0, argument1, argument2, argument3, argument4, argument5, argument6, argument7) {

	var inst = instance_create_depth(argument0, argument1, -10000, obj_lineGraph);

	with (inst)
		{
		width = argument2;
		height = argument3;
		xAxisIdentifier = argument4;
		xAxisIncrementStep = argument5;
		yAxisIdentifier = argument6;
		title = argument7;
		}

	scr_lineGraphRecalculate(inst);
	
	return (inst);


}
