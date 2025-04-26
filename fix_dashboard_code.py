import os
import sys

# Path to the visualization_dashboard.py file
dashboard_file = 'src/visualization/visualization_dashboard.py'

# Check if file exists
if not os.path.exists(dashboard_file):
    print(f"Error: File not found at {dashboard_file}")
    sys.exit(1)

# Read the current content
with open(dashboard_file, 'r') as f:
    content = f.read()

# Check if the file already has the app.run instead of app.run_server
if 'app.run_server(' in content:
    # Replace app.run_server with app.run
    updated_content = content.replace(
        'app.run_server(port=port, debug=debug)', 
        'app.run(port=port, debug=debug, host=\'0.0.0.0\')'
    )
    
    # Write the updated content back to the file
    with open(dashboard_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {dashboard_file}")
    print("The dashboard should now work with main.py")
else:
    if 'app.run(' in content:
        print(f"File already contains app.run - no changes needed")
    else:
        print(f"Could not find 'app.run_server' in the file to replace")
        print("Manual inspection required")