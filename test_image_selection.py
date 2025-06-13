#!/usr/bin/env python3
"""
Test image selection functionality
"""

import tkinter as tk
from tkinter import filedialog
import os

def test_file_dialog():
    """Test if file dialog works"""
    try:
        print("Testing file dialog...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        img_path = filedialog.askopenfilename(
            title="Select a test image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if img_path:
            print(f"✅ File dialog works! Selected: {os.path.basename(img_path)}")
            print(f"Full path: {img_path}")
            return True
        else:
            print("❌ No file selected")
            return False
            
    except Exception as e:
        print(f"❌ Error with file dialog: {e}")
        return False
    finally:
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    print("=== Testing Image Selection ===")
    test_file_dialog()
    print("Test complete!")
