# PyQt5 Joystick Control GUI

This is a PyQt5-based GUI application for robot control with joystick integration and user authentication.

## Features

- **User Authentication**: SQLite-based login system with role-based access control
- **Joystick Integration**: Real-time joystick input visualization using pygame
- **Role-Based UI**: Different interface panels based on user roles (owner/operator)
- **Visual Feedback**: Dynamic image switching (brown/orange) based on joystick input

## Installation

### Prerequisites

Ensure you have Python 3 and the following packages installed:

```bash
pip install PyQt5 pygame
```

### Database Setup

Run the database setup script to create the users database:

```bash
cd gui
python3 setup_database.py
```

This will create `users.db` with two default users:

| Username   | Password  | Role     |
|------------|-----------|----------|
| developer  | 12121212  | owner    |
| operator   | 12121212  | operator |

## Running the Application

```bash
python3 gui/gui.py
```

Or from within the gui directory:

```bash
cd gui
python3 gui.py
```

## Usage

### Login

1. When the application starts, you'll see the login tab
2. Enter username and password
3. Click "Enviar" (Submit) to login

### Operation Tab

After successful login, the operation tab will be displayed with:

- **Left Panel (Joystick)**: Shows joystick directional input with visual feedback
  - Brown arrows: Default state (no input)
  - Orange arrows: Active state (joystick pressed)

- **Right Panel (Camera)**: Reserved for camera feed display

- **Owner Panel** (only visible for 'owner' role): Displays body position and orientation data
  - Position: X, Y, Z coordinates
  - Orientation: Roll, Pitch, Yaw

### Joystick Controls

The application supports standard USB game controllers:
- **D-Pad/Hat buttons**: Primary directional input
- **Left Analog Stick**: Alternative directional input (threshold: 50%)

The arrow images will change from brown to orange when the corresponding direction is pressed.

## Project Structure

```
gui/
├── gui.py              # Main application file
├── untitled.ui         # PyQt5 UI definition file
├── setup_database.py   # Database initialization script
├── users.db            # SQLite database (created after setup)
├── README.md           # This file
└── icons/              # Image assets
    ├── up_brown.png
    ├── up_orange.png
    ├── down_brown.png
    ├── down_orange.png
    ├── left_brown.png
    ├── left_orange.png
    ├── right_brown.png
    └── right_orange.png
```

## Key Components

### Authentication System
- Database: SQLite (`users.db`)
- Password: Plain text (consider hashing for production)
- Roles: owner, operator

### Joystick Integration
- Library: pygame
- Polling rate: 50ms (20 Hz)
- Supports both D-pad/hat and analog stick input

### UI Features
- Tab-based navigation
- Dynamic panel visibility based on user role
- Real-time image updates based on joystick state

## Troubleshooting

### No Joystick Detected
If you see "No joystick detected" in the console:
- Ensure your controller is connected
- Check if the controller is recognized by your system
- Try unplugging and replugging the controller
- The application will continue to work without joystick support

### Joystick Permission Issues
If you see permission errors like "Unable to open /dev/input/eventXX":
- The application will gracefully handle this and continue without joystick support
- To fix, add your user to the `input` group:
  ```bash
  sudo usermod -a -G input $USER
  ```
- Then log out and log back in

### Login Failed
- Verify the database exists: `ls -l gui/users.db`
- Re-run the setup script if needed
- Check credentials match the default users

## Customization

### Adding New Users

You can add new users by modifying `setup_database.py` or directly through SQLite:

```python
import sqlite3
conn = sqlite3.connect('gui/users.db')
cursor = conn.cursor()
cursor.execute(
    'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
    ('newuser', 'password', 'operator')
)
conn.commit()
conn.close()
```

### Modifying UI

The UI layout is defined in `untitled.ui` and can be edited using:
- Qt Designer (GUI tool)
- Direct XML editing
- PyQt5 code in `gui.py`

## Notes

- The application runs in a single window with tab-based navigation
- Login tab is disabled after successful authentication
- Owner panel visibility is controlled by user role
- Images are scaled to fit labels while maintaining aspect ratio
