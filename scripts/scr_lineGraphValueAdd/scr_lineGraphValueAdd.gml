/// @description scr_lineGraphValueAdd(id, value);
/// @param id
/// @param value
function scr_lineGraphValueAdd(argument0, argument1) {

	var instID = argument0;
	var val = argument1;


	if (instance_exists(instID)) // Verify that instance exists.
		{
		with (instID)
			{
			if (object_index == obj_lineGraph) // Verify that object_id matches.
				{
				var listSize = ds_list_size(values);
			
				if (listSize > maxValues)
					{
					ds_list_delete(values, 0);
					}
			
				ds_list_add(values, val); // Add value to internal list.
	
				if (val > yAxisHighestValue) // Check if new value is larger than the largest recorded value.
					{
					yAxisHighestValue = val; // Set the highest recorded value to the new value.
					}
	
			
				var maxStartPos = floor(listSize - (defaultVisibleDataPoints / zoom));
				if (startPos < maxStartPos)
					{
					startPos = clamp(maxStartPos - 1, 0, listSize);
					}
				valueCount++;
				valueSum += val;
				}
			else
				{
				show_debug_message("Error: 'scr_lineGraphValueAdd()' failed due to incorrect object type.");
				}
			}
		}
	else
		{
		show_debug_message("Error: 'scr_lineGraphValueAdd()' failed, instance does not exist.");
		}


}
