/// @description

enum uiInputState
	{
	none,
	hover,
	pressed
	}
	
height = 128; // Total height of the frame.
width = 384; // Total width of the frame.
padding = 2; // Padding

dataSurfWidth = width - (padding * 2); // Surface Width.
dataSurfHeight = height - (padding * 2); // Surface Height.

frameSurf = -1; // Surface containing the entire ui element.
graphSurf = -1; // Surface for drawing the graph data.

title = "Graph Title"; // Overall title of the graph. (eg. "Incoming Packets")

zoom = 1.0; // Current zoom. (scalar value)
zoomMax = 4.0; // Maximum zoom. (scalar value)
zoomMin = 0.25; // Minimum zoom. (scalar value)

colour = make_colour_hsv(irandom(255), 180, 255); // Graph data colour.

graphDirection = 1; // Graph direction; (1 = new data appears on the right side, -1 = left)

yAxisIdentifier = "Quantity"; // Name of the y axis. (what it's measuring)
yAxisHighestValue = 2; // Highest recorded value on the y axis. (The grid will adjust if this number is surpassed)
yAxisTickCount = 5; // How many increment ticks should divide the max recorded number on the y axis.
xAxisIdentifier = "Milliseconds"; // Name of the x axis. (what it's measuring)
xAxisIncrementStep = (1 / room_speed) * 1000; // Size of value increments on the x axis. 

showPoints = 1; // Displays a circle around data points on the graph.
showLines = 1; // Displays lines from one data point to the next. (Not currently used)
pointRadius = 2; // The radius of points when points are shown.
lineThickness = 4; // Determines line thickness when lines are shown. (Not currently used)

#region Value vars
maxValues = 200; // Maximum number of stored values.
defaultVisibleDataPoints = 32;
startPos = 0; // The minimum value index to process. (scroll position)(index value)
valueCount = 0; // The total number of values that have been added to the graph.
valueSum = 0; // The total sum of values that have been added to the graph.
values = ds_list_create(); // list of stored values.
#endregion Value vars

#region General UI variables
dt = 0; // Delta time. (seconds)
xPressed = 0;
yPressed = 0;
focus = 0; // Whether or not the ui element has focus. (not currently used)
inputState = 0; // Current input state; (0 = none, 1 = hover, 2 = pressed)
allowInput = 1; // Whether to allow mouse interactivity. (bool)
frameVisible = 1; // Whether or not the frame is visible. (also disables input)(bool)
frameFadeProgress = 0; // The current fade progress of the frame (scalar value; 0.0 = fully visible, 1.0 = fully faded)
frameFadeRate = 0.5; // How long (in seconds) the frame should take to fade.
frameFadeXOffset = 0; // Determines how far the panel will shift on the x axis when full faded. (negative values will invert the effect)
frameFadeYOffset = height; // Determines how far the panel will shift on the y axis when full faded. (negative values will invert the effect)
#endregion General UI variables


