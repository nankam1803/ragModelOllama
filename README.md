📦 Project Setup & Installation

This guide will help you quickly set up your Python project, including creating a virtual environment, installing dependencies, and running the application.

✅ Step 1: Clone the Repository

Clone the repository from GitHub:

git clone <your-repo-url>
cd <your-repo-name>

Replace <your-repo-url> and <your-repo-name> with your actual repository information.

✅ Step 2: Create and Activate a Virtual Environment

Create a virtual environment named .venv:

Windows:

python -m venv .venv
.\.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

✅ Step 3: Install Dependencies

Install all the required Python libraries using the provided requirements.txt file:

pip install --upgrade pip
pip install -r requirements.txt

✅ Step 4: Run the Application

Run your Python script or application:

python ragModel.py

Make sure to replace ragModel.py with your actual Python script filename.

🚩 Troubleshooting

If you encounter errors related to Poppler (common when working with PDFs), follow these additional steps:

Windows Users:

Download Poppler.

Extract it to a folder (e.g., C:\poppler).

Add C:\poppler\Library\bin to your system PATH.

macOS Users:

brew install poppler

Linux Users (Ubuntu/Debian):

sudo apt-get install poppler-utils

Then, restart your terminal and re-run your application.