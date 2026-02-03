###########################################################################################################
## VARIABLES
###########################################################################################################
PYTHON=python
CONDA=conda
CURRENT_DIR := $(PWD)
SRC_DIR=$(CURRENT_DIR)/src
MAIN_DIR=$(SRC_DIR)/main

###########################################################################################################
## SCRIPTS
###########################################################################################################

# Create conda env to run MM
create-env:
	$(CONDA) create --name ml-algo-scratch python=3.11

update-env:
	$(CONDA) env update --file environment.yml

# Export Conda Environment
conda-export-env:
	$(PYTHON) conda_export_minimal.py --s_save="environment.yml"

# Run Linear Regression
run-linear-regression:
	$(PYTHON) -m src.main.main_linear_regression \
		--n_samples $(N_SAMPLES) \
		--n_features $(N_FEATURES) \
		--noise $(NOISE) \
		--random_state $(RANDOM_STATE) \
		--test_size $(TEST_SIZE) \
		--learning_rate $(LEARNING_RATE) \
		--n_iters $(N_ITERS) \
		--optimization_metric $(OPTIMIZATION_METRIC) \
		$(VERBOSE)

# Run KNN
run-knn:
	$(PYTHON) -m src.main.main_knn \
		--n_samples $(N_SAMPLES) \
		--n_features $(N_FEATURES) \
		--n_classes $(N_CLASSES) \
		--n_informative $(N_INFORMATIVE) \
		--random_state $(RANDOM_STATE) \
		--test_size $(TEST_SIZE) \
		--k $(K) \
		$(VERBOSE)
