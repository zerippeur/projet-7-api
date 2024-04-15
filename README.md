# PREDICTION APP (FASTAPI)
### <i>Projet réalisé dans le cadre de la formation 'Data Scientist' d'Openclassrooms</i>

Liens utiles :   
- Compétition Kaggle à l'origine du projet : <a>https://www.kaggle.com/c/home-credit-default-risk</a>
- Lien vers les données : <a>https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip</a>
- Feature engineering utilisé : <a>https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features</a>

***

### Content of the folder

  This folder contains the fastapi prediction app's project.
  The structure is a regular poetry structure with the following folders:
  - .github/workflows: folder with the cicd yaml file for github actions
  - .venv: virtual environement setp up with poetry
  - api: project folder for the app with the following files:
    - app.py: script for the fastapi app
  - tests: folder for tests (three unitary tests concerning the prediction results fed to the dashboard)

  Additional files are used for deployment:
  - Procfile: file for command run to launch the fastapi app once deployed
  - poetry.lock: file generated automatically by poetry
  - pyproject.toml: file with the dependencies
  - requirements.txt: file for the build during deployment (generated by poetry)
  - runtime.txt: file to specify python version for the build during deployment
  - README.md
  - other files related to the dev environment (.gitignore, .flake8...)

***

### Project

  This project is a fastapi prediction app that feeds a streamlit dashboard. The app returns model's prediction for a given client as well as global (built-in or SHAP) and local (SHAP) feature importance.

  The deployed dashboard can be found here:
  <a>https://oc-project-7-dashboard-a5355cc087c9.herokuapp.com/</a>

  And the prediction api's docs:
  <a>https://oc-project-7-api-2592e43c0c28.herokuapp.com/docs#/</a>