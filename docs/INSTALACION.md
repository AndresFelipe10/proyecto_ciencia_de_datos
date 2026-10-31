# Instalación y Configuración

## Paquetes del sistema

Instala los paquetes necesarios siguiendo [estas instrucciones para Ubuntu 26.04](https://github.com/clubdecomputacion/lab-files/blob/main/archivos/ubuntu_setup.md)

```bash
python3 --version
git --version
gh --version
make --version
```

## Fork/Clone del repositorio 

Abre una Terminal:

```bash
mkdir -p ~/proyectos && cd ~/proyectos
gh repo fork https://github.com/dfmarin/proyecto_ciencia_de_datos --clone
cd proyecto_ciencia_de_datos     # entra a la carpeta clonada
```

## Recibir actualizaciones del profesor

Configura `upstream`:

```bash
git remote add upstream https://github.com/dfmarin/proyecto_ciencia_de_datos.git
git remote -v
```

Recibe las actualizaciones:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## Entorno virtual e instalación de dependencias

Desde la raíz del proyecto, debe contener el archivo `pyproject.toml` :

```bash
python3 -m venv .venv
source .venv/bin/activate 
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verifica la instalación:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn, pyarrow, joblib, dotenv; print('OK')"
pip list | grep -Ei "pandas|numpy|scikit|seaborn|scipy|pyarrow|pytest|ruff|jupyter"
```

Congelar versiones exactas:

```bash
pip freeze > requirements.lock.txt
```

> Cada vez que abras una terminal nueva: `cd ~/proyectos/proyecto_ciencia_de_datos && source .venv/bin/activate`.

## Configuración local (`.env`)

```bash
cp .env.example .env
cat .env
```

Valores por defecto: `TITANIC_URL`, `RANDOM_STATE=42`, `TEST_SIZE=0.2`, `CV_FOLDS=5`. El `.env` **no se versiona**.

