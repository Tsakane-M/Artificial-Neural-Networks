VENV := venv

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/bin/activate

run_Classifier: venv
	./$(VENV)/bin/python3 Classifier.py 
	

run_XOR: venv
	./$(VENV)/bin/python3 XOR.py 


clean:
	rm -rf $(VENV) __pycache__
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean




