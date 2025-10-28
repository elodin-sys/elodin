# Multi-Window Feature Usage Guide

## Opening the Command Palette

To access the multi-window feature:
1. Press **Cmd+P** (on macOS) or **Ctrl+P** (on Linux/Windows) to open the command palette
2. Type "Add New Window" or just start typing "window" to filter commands
3. Press Enter to select the command

## Window Options

When you select "Add New Window", you'll see two options:

### 1. Empty Viewport
Creates a new window with an empty 3D viewport that you can navigate independently. This is useful for:
- Viewing the simulation from multiple angles simultaneously
- Having a dedicated camera view for specific objects
- Monitoring different parts of your simulation

### 2. Current Tab
Pops out the currently active tab to a new window. This moves the selected pane from the main editor to a secondary window. Note that:
- The pane will be removed from the main window
- Currently, there's no way to return it to the main window (you'll need to recreate it)

## Keyboard Shortcuts

- **Cmd/Ctrl + P**: Open command palette
- **Up/Down arrows**: Navigate through command options
- **Enter**: Execute selected command
- **Escape**: Close command palette

## Current Limitations

1. **UI Panels**: Due to `bevy_egui` limitations, UI-heavy panels (graphs, monitors, inspector) won't render correctly in secondary windows. Only 3D viewports display properly.

2. **One-way Operation**: Once a pane is moved to a secondary window, it cannot be returned to the main window. You'll need to close the secondary window and recreate the pane in the main window if needed.

3. **State Synchronization**: Time, selection, and playback states are not fully synchronized between windows yet.

## Best Use Cases

The multi-window feature works best for:
- **3D Viewports**: Multiple camera angles of your simulation
- **Dedicated monitoring**: Having a separate window focused on a specific aspect
- **Multi-monitor setups**: Spreading your workspace across multiple displays

## Tips

- The command palette remembers your recent searches, so you can quickly access frequently used commands
- You can start typing immediately after opening the palette - no need to click in the search field
- The palette shows keyboard shortcuts for commands that have them

## Troubleshooting

If a secondary window doesn't appear:
1. Check if it's behind the main window
2. Look for it in your OS's window switcher (Alt+Tab on Windows/Linux, Cmd+Tab on macOS)
3. Check the terminal output for any error messages

If a pane doesn't render correctly in a secondary window:
- This is expected for UI panels due to current technical limitations
- Use secondary windows primarily for 3D viewports

## Future Improvements

We're working on:
- Full UI panel support in secondary windows
- Ability to return panes to the main window
- Better state synchronization
- Window position and size persistence
- Drag-and-drop between windows
