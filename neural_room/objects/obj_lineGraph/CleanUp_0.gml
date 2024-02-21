/// @description
if (ds_exists(values, ds_type_list)) 
	{
	ds_list_destroy(values);
	} 
if (surface_exists(graphSurf)) surface_free(graphSurf);
if (surface_exists(frameSurf)) surface_free(frameSurf);