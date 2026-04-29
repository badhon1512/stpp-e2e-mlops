from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TEMP_DIR = ARTIFACTS_DIR / "tmp"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_FILE = LOG_DIR / "inference.log"

PROJECT_NAME = "support-ticket-priority-platform"
PROJECT_DISPLAY_NAME = "Support Ticket Priority Classification Platform"

DATASET_NAME = "tobiasbueck/multilingual-customer-support-tickets"
DEFAULT_LANGUAGE = "en"
RAW_DATA_FILE = RAW_DATA_DIR / "dataset-tickets-multi-lang-4-20k.csv"
TEXT_FIELDS = ("subject", "body")
TARGET_COLUMN = "priority"
LABELS = ("low", "medium", "high")
TEXT_COLUMN = "text"
LANGUAGE_COLUMN = "language"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
DATASET_PROFILE_FILE = REPORTS_DIR / "dataset_profile.json"
EDA_REPORT_FILE = REPORTS_DIR / "eda_report.json"
VALIDATION_REPORT_FILE = REPORTS_DIR / "validation_report.json"
TRAINING_REPORT_FILE = REPORTS_DIR / "training_report.json"
LOCAL_MODEL_FILE = MODELS_DIR / "ticket_priority_sklearn.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

REGISTERED_MODEL_NAME = "ticket-priority-sklearn"
MODEL_ALIAS = "champion"
EXPERIMENT_NAME = "ticket-priority-classification"

MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)
MAX_ITER = 1000
C_VALUE = 4.0

APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_RELOAD = True
