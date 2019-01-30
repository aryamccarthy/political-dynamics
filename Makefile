.PHONY: clean data lint requirements interim

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = political-dynamics
PYTHON_INTERPRETER = python3

UNZIP = unzip -uD

RAW = data/raw
INTERIM = data/interim
PROCESSED = data/processed
CODINGS = data/codings
FIGURES = reports/figures

YEARS = 1988 1992 1996 2000 2004 2008 2012 2016

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

$(FIGURES)/correlations_%.pdf: $(PROCESSED)/%.csv
	$(PYTHON_INTERPRETER) src/visualization/spearman.py --infile $^ --outfile $@

## Make Dataset
data: requirements processed

processed: $(addprefix $(PROCESSED)/,$(addsuffix .csv,$(YEARS)))  

$(PROCESSED)/%.csv: $(INTERIM)/%.dta $(CODINGS)/%.tsv
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $^ $@

##$(PROCESSED)/%.csv:	$(INTERIM)/%.dta src/data/make_dataset.py
##	$(PYTHON_INTERPRETER) src/data/make_dataset.py	

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## All interim data
interim: $(addprefix $(INTERIM)/,$(addsuffix .dta,$(YEARS)))  

## 1988 interim data
$(INTERIM)/1988.dta: $(RAW)/anes1988dta.zip
	$(UNZIP) -j $< NES1988.dta -d $(dir $<); \
	mv $(dir $<)NES1988.dta $@

## 1992 interim data
$(INTERIM)/1992.dta: $(RAW)/anes1992dta.zip
	$(UNZIP) -j $< NES1992.dta -d $(dir $<); \
	mv $(dir $<)NES1992.dta $@

## 1996 interim data
$(INTERIM)/1996.dta: $(RAW)/anes1996dta.zip
	$(UNZIP) -j $< nes96.dta -d $(dir $<); \
	mv $(dir $<)nes96.dta $@

## 2000 interim data
$(INTERIM)/2000.dta: $(RAW)/anes2000TSdta.zip
	$(UNZIP) -j $< anes2000TS.dta -d $(dir $<); \
	mv $(dir $<)anes2000TS.dta $@

## 2004 interim data
$(INTERIM)/2004.dta: $(RAW)/anes2004TSdta.zip
	$(UNZIP) -j $< anes2004TS.dta -d $(dir $<); \
	mv $(dir $<)anes2004TS.dta $@

## 2008 interim data
$(INTERIM)/2008.dta: $(RAW)/anes_timeseries_2008_dta.zip
	$(UNZIP) -j $< anes_timeseries_2008.dta -d $(dir $<); \
	mv $(dir $<)anes_timeseries_2008.dta $@

## 2012 interim data
$(INTERIM)/2012.dta: $(RAW)/anes_timeseries_2012_dta.zip
	$(UNZIP) -j $< anes_timeseries_2012.dta -d $(dir $<); \
	mv $(dir $<)anes_timeseries_2012.dta $@

## 2016 interim data
$(INTERIM)/2016.dta: $(RAW)/anes_timeseries_2016_dta.zip
	$(UNZIP) -j $< anes_timeseries_2016.dta -d $(dir $<); \
	mv $(dir $<)anes_timeseries_2016.dta $@

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
