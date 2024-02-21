/// @description

dt = delta_time * 0.000001; // Convert delta_time to seconds

var listSize = ds_list_size(values);

if (frameVisible < 1)
	{
	frameFadeProgress = min(1.0, frameFadeProgress + (dt / frameFadeRate));
	}
else
	{
	frameFadeProgress = max(0.0, frameFadeProgress - (dt / frameFadeRate));
	}

#region Input
if ((allowInput) && (frameVisible))
	{
	if (inputState < uiInputState.pressed)
		{
		inputState = scr_mouseCheckBoxGUI(x, x + width, y,  y + height);
		if (inputState == uiInputState.hover)
			{
			if (mouse_check_button_pressed(mb_left))
				{
				inputState = uiInputState.pressed;
				xPressed = window_mouse_get_x() - x;
				yPressed = window_mouse_get_y() - y;
				}
			var zoomInput = (mouse_wheel_up() - mouse_wheel_down());
			if (zoomInput != 0)
				{
				zoom += zoomInput * 0.1;
				zoom = clamp(zoom, zoomMin, zoomMax);
				}
			if (listSize > 0) 
				{
				var nudge = keyboard_check(ord("D")) - keyboard_check(ord("A"));
				startPos = clamp(startPos + nudge, 0, max(0,floor(listSize - (defaultVisibleDataPoints / zoom) - 1)));
				}
			}
		}
	else
		{
		if (mouse_check_button_released(mb_left))
			{
			inputState = uiInputState.none;		
			}
		x = window_mouse_get_x() - xPressed;
		y = window_mouse_get_y() - yPressed;
		}
	}
#endregion
