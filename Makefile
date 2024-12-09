# Подготовка виртуального окружения
.PHONY: venv
venv:
	python3 -m venv ./.venv

# Установка указанных пакетов (когда отсутствует файл `requirements.txt`)
.PHONY: direct_deps
direct_deps: venv
	. ./.venv/bin/activate
	pip install jupyter numpy pandas matplotlib seaborn torch transformers datasets python-dotenv accelerate protobuf sentencepiece peft

# Установка пакетов указанных в `requirements.txt`
.PHONY: deps
deps: venv
	. ./.venv/bin/activate
	pip install -r requirements.txt

# Удаление виртуального окружения
.PHONY: del_venv
del_venv:
	rm -R ./.venv

# Сохранение зависимостей в файл `requirements.txt`
.PHONY: freeze
freeze:
	. ./.venv/bin/activate
	pip freeze > requirements.txt