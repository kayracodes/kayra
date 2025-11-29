// Function to open the new window
function openGoToGoalWindow() {
    // Create a new window
    const newWindow = window.open('', 'Go To Goal', 'width=300,height=300');
    
    // Create the content for the new window
    const content = `
        <html>
        <head>
            <title>Go To Goal</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                input, button { margin: 5px 0; }
                #feedback { color: #A00000; margin-top: 10px; }
            </style>
        </head>
        <body>
            <p>Enter the latitude and longitude</p>
            <input type="number" id="lat" placeholder="Latitude" step="any">
            <input type="number" id="long" placeholder="Longitude" step="any">
            <br>
            <button onclick="submitGoal()">Enter</button>
            <button onclick="window.close()">Cancel</button>
            <div id="feedback"></div>

            <script>
                function submitGoal() {
                    const latiGoal = document.getElementById('lat').value;
                    const longiGoal = document.getElementById('long').value;
                    const latiCurrent = ${gps_client.lat};
                    const longiCurrent = ${gps_client.long};

                    if (latiGoal === '' || longiGoal === '') {
                        document.getElementById('feedback').innerText = 'Please enter both latitude and longitude.';
                    } else {
                        // Call the parent window's go_to_goal function
                        window.opener.go_to_goal(parseFloat(latiCurrent), parseFloat(longiCurrent), parseFloat(latiGoal), parseFloat(longiGoal));
                        
                        window.close();
                    }
                }
            </script>
        </body>
        </html>
    `;
    
    // Write the content to the new window
    newWindow.document.write(content);
}

// The main go_to_goal function
function go_to_goal(latiCurrent, longiCurrent, latiGoal, longiGoal) {
    // Here you can implement the logic to handle the lat and long values
    // For now, we'll just log them to the console
    console.log(longiGoal); 
    console.log(`Going to goal: Latitude ${latiGoal}, Longitude ${longiGoal}`);

    var GoalServiceClient = new ROSLIB.Service({
        ros: ros,
        name: "/go_to_gps",
        serviceType: "outdoor_waypoint_nav/GoToGps"
    })

    var request = new ROSLIB.ServiceRequest({
        latiCurrent: latiCurrent ,
        longiCurrent: longiCurrent,
        latiGoal: latiGoal,
        longiGoal: longiGoal
    })

    GoalServiceClient.callService(request, function(result){
        relative_x = result.target_pose.pose.position.x;
        relative_y = result.target_pose.pose.position.y;

        console.log("Have gone " + relative_x + " on x" + relative_y + " on y");
    })
    // You can add your external function calls here
}