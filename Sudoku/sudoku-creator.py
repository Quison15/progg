import tkinter as tk
from tkinter import messagebox

def save_sudoku():
    with open("sudoku_output.txt", "w") as f:
        entries = []
        for row in range(9):
            for col in range(9):
                value = grid[row][col].get()
                if value.isdigit() and 1 <= int(value) <= 9:
                    entries.append(f"({value}, {row + 1}, {col + 1})")
        f.write(",\n".join(entries))
    messagebox.showinfo("Saved", "Sudoku grid saved to sudoku_output.txt")

def create_sudoku_gui():
    global grid
    root = tk.Tk()
    root.title("Sudoku Creator")
    
    grid = []
    for i in range(9):
        row_entries = []
        for j in range(9):
            entry = tk.Entry(root, width=5, font=("Arial", 18), justify="center")
            entry.grid(row=i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        grid.append(row_entries)
    
    save_button = tk.Button(root, text="Save", command=save_sudoku, font=("Arial", 14))
    save_button.grid(row=9, column=0, columnspan=9, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_sudoku_gui()