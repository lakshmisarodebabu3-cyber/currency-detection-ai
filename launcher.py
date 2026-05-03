import os
import sys

# ---------------- CONFIG ----------------
# Map choices to files
modes = {
    "1": {
        "name": "Mouse Detection",
        "file": "mouse_ui.py"
    },
    "2": {
        "name": "Keyboard Detection",
        "file": "keyboard_ui.py"
    },
    "3": {
        "name": "Webcam Detection",
        "file": "pyqt_currency_webcam.py"
    },
    "4": {
        "name": "Multi-Currency Detection",
        "file": "pyqt_currency_multi.py"
    },
    "5": {
    "name": "Predict",
    "file": "predict.py"
    }


}

# ---------------- FUNCTIONS ----------------
def print_menu():
    print("\033[1;34m================ Currency Detection Launcher ================\033[0m")
    for key, mode in modes.items():
        print(f"\033[1;32m{key}\033[0m: {mode['name']}")
    print("Q: Quit")
    print("\033[1;34m===========================================================\033[0m")

def run_mode(choice):
    file_path = modes[choice]["file"]
    if not os.path.exists(file_path):
        print(f"\033[1;31mError: {file_path} not found!\033[0m")
        return
    print(f"\033[1;33mRunning {modes[choice]['name']}...\033[0m")
    os.system(f"python {file_path}")  # Runs the selected Python file

# ---------------- MAIN ----------------
if __name__ == "__main__":
    while True:
        print_menu()
        choice = input("\033[1;36mEnter your choice: \033[0m").strip().lower()
        if choice == "q":
            print("\033[1;31mExiting launcher. Goodbye!\033[0m")
            sys.exit(0)
        elif choice in modes:
            run_mode(choice)
        else:
            print("\033[1;31mInvalid choice. Try again.\033[0m")