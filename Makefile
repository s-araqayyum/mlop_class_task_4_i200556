setup:
	python -m venv venv
	.venv/bin/activate; pip install -r requirements.txt

run:
	. venv/bin/activate; FLASK_APP=app.py FLASK_ENV=development flask run

docker-build:
	docker build -t wine_quality_prediction .

docker-run:
	docker run -p 5000:5000 wine_quality_prediction

