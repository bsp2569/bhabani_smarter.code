# Webpage Content Search

## Project Structure
root/
│── backend/
│     ├── app.py
│     ├── requirements.txt
│     └── utils/
│
│── frontend/
│     ├── page.tsx
│     ├── package.json
│     ├── components/
│     └── styles/

## System Requirements
	•	Python 3.10+
	•	Node.js 18+
	•	npm or yarn
	•	Git

***Set front-end and backend in different terminals*** <br>

## Backend:
**Create Virtual Environment**
	
	cd backend
	python3 -m venv venv
	source venv/bin/activate    # mac
	venv\Scripts\activate       # Windows
	

**Install dependencies present in requirements.txt file** <br>
    
	pip install -r requirements.txt

**Run the flask app**

	python app.py

## Frontend
**Install dependencies**
	
	cd frontend
	npm install 

**Start running**
	
	npm run dev
*The server wil start running at http://localhost:3000*


