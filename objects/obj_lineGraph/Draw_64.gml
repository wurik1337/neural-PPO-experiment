/// @description
var listSize = ds_list_size(values);
var visTicks = ceil(defaultVisibleDataPoints / zoom) + 1;
var xGridSkip = max(1, (round((defaultVisibleDataPoints / zoom) / 32)));

var xBase = dataSurfWidth * 0.5;
xBase += xBase * -graphDirection;
var xInc = ((dataSurfWidth / defaultVisibleDataPoints) * zoom) * graphDirection;
var yInc = dataSurfHeight / yAxisTickCount;
var drawGraphSurf = 0;

if (listSize > 0)
	{
	if (!surface_exists(graphSurf))
		{
		graphSurf = surface_create(dataSurfWidth, dataSurfHeight);
		}

	surface_set_target(graphSurf); // Set surface target.
	draw_clear_alpha(c_dkgray, 0); // Clear surface.
	var cx = 0;
	var cy = 0;
	#region Draw horizontal grid lines
	draw_set_alpha(0.5);
	draw_set_colour(c_white);
	for (var i = 0; i < yAxisTickCount; i++) // Draw y Axis Grid Lines (Horizontal)
		{
		draw_rectangle(0, i * yInc, width, (i * yInc) + 1, 0);
		}
	#endregion Draw horizontal grid lines
	
	#region Primitive setup
	draw_set_alpha(1);
	draw_set_colour(colour);
	draw_set_alpha(0.8);
	draw_primitive_begin(pr_trianglestrip);
	#endregion Primitive Setup
	
	var p = values[| startPos] / yAxisHighestValue;
	cx = xBase + (-startPos * xInc);
	cy = dataSurfHeight - (dataSurfHeight * p);
	draw_vertex(cx, cy);
	draw_vertex(cx, dataSurfHeight);
	var endPos = clamp(startPos + visTicks, 0, listSize);
	for (var i = startPos; i < endPos; i++)
		{
		var val = values[| i];
		#region Draw vertical grid lines
		if ((i mod xGridSkip == 0))
			{
			draw_set_alpha(0.5);
			draw_set_colour(c_white);
			draw_rectangle(xBase + ((i - startPos) * xInc), 0, xBase + ((i - startPos) * xInc) + 1, height, 0);
			}
		#endregion Draw vertical grid lines
		
		#region Draw vertices
		draw_set_colour(colour);
		draw_set_alpha(0.8);
		p = val / yAxisHighestValue;
		cx = xBase + ((i - startPos) * xInc);
		cy = dataSurfHeight - (dataSurfHeight * p);
		draw_vertex(cx, cy);
		draw_vertex(cx, dataSurfHeight);
		#endregion Draw vertices
		
		#region Draw points
		draw_set_alpha(1);
		if (showPoints)
			{
			draw_circle(cx, cy, pointRadius, 0);
			}
		#endregion Draw Points
		}
	draw_primitive_end();
	surface_reset_target();

	drawGraphSurf = 1;
	}

//draw_set_font(fnt_debug);
draw_set_alpha(1);
draw_set_colour(c_white);
draw_set_halign(fa_center);


if (!surface_exists(frameSurf))
	{
	frameSurf = surface_create(width, height);
	}

surface_set_target(frameSurf); // Set surface target.

scr_uiDrawElementBox(0, 0, width, height, 2); // Draw frame background.

gpu_set_colorwriteenable(1,1,1,0);

if (drawGraphSurf)
	{
	draw_surface(graphSurf, padding, padding);
	}

#region Indicate no data if no values exist
if (listSize < 1)
	{
	draw_set_valign(fa_middle);
	draw_text(width * 0.5, height * 0.5, "NO DATA");
	}
#endregion Indicate no data if no values exist

#region Draw Graph Title Label
draw_set_valign(fa_top);
draw_text(width * 0.5, padding, title);
#endregion

#region Draw xAxis Label
draw_set_valign(fa_bottom);
draw_text(width * 0.5, height, xAxisIdentifier);
#endregion

#region Draw yAxis Label
draw_set_halign(fa_right);
draw_set_valign(fa_top);
var charH = string_height("w\nw") * 0.45;
var strLength = string_length(yAxisIdentifier);
var strHeight = strLength * charH;
var yBase = (height * 0.5) - (strHeight * 0.5);
for (var i = 0; i < strLength; i++)
	{
	draw_text(width - padding, yBase + (i * charH), string_char_at(yAxisIdentifier, i + 1)); 
	}
#endregion
	
#region Draw Tick Markers
draw_set_alpha(0.6);
if (graphDirection > 0)
	{
	draw_set_halign(fa_left);
	xBase = padding;
	}
else
	{
	draw_set_halign(fa_right);
	xBase = width - padding;
	}

draw_set_valign(fa_bottom);
//draw_set_font(fnt_debugSm);
draw_text(xBase, height, string_format(xAxisIncrementStep * xGridSkip, 0, 1));

draw_set_halign(fa_right);
draw_set_valign(fa_top);

var incSize = yAxisHighestValue / yAxisTickCount;
yBase = height - (padding * 2);
xBase = (width - padding) - 18;
for (var i = 1; i < yAxisTickCount + 1; i++) // Draw y Axis Grid Tick Values. (Horizontal)
	{
	draw_text(xBase, yBase - (i * yInc), string_format(incSize * i, 0, 1));
	}
draw_set_alpha(1);
#endregion

#region Draw Average
if (valueCount > 0)
	{
	draw_set_halign(fa_left);
	draw_text(padding, padding, "Avg: " + string(valueSum / valueCount));
	}
#endregion

#region Draw Hover Highlight
gpu_set_blendmode(bm_add);
draw_set_alpha(0.4);
if (inputState > uiInputState.none)
	{
	draw_sprite_stretched(spr_uiHighlight, 0, 0, 0, width, height);
	}
gpu_set_blendmode(bm_normal);
draw_set_alpha(1);
#endregion

gpu_set_colorwriteenable(1,1,1,1);
surface_reset_target();

var p = scr_smoothStep(0, 1, frameFadeProgress); // Add smoothing

draw_surface_ext(frameSurf, x + (frameFadeXOffset * p), y + (frameFadeYOffset * p), 1, 1, 0, -1, 1 - p); // Draw frame
