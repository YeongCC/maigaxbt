from __future__ import annotations

from os import getenv

from api.config.silk import SILKY_MIDDLEWARE_CLASS, USE_SILK

PROJECT_NAME = getenv("PROJECT_NAME", "django_template")
PROJECT_VERBOSE_NAME = getenv("PROJECT_VERBOSE_NAME", "Django Template")

ENVIRONMENT = getenv("ENVIRONMENT", "local")
HOST = getenv("HOST", "localhost")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "axes",
    "silk",
    "rest_framework",
    "drf_spectacular",
    "api.user.apps.UserConfig",
    "api.analysis.apps.AnalysisConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    SILKY_MIDDLEWARE_CLASS,
    "axes.middleware.AxesMiddleware",
]

if not USE_SILK:
    INSTALLED_APPS.remove("silk")
    MIDDLEWARE.remove(SILKY_MIDDLEWARE_CLASS)

ROOT_URLCONF = "api.web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "api.web.wsgi.application"

LANGUAGE_CODE = getenv("LANGUAGE_CODE", "en-us")

USE_TZ = True
TIME_ZONE = getenv("TIME_ZONE", "UTC")

USE_I18N = True

GPT_API_KEY = getenv("GPT_API_KEY")

CHART_API_HOST = getenv("CHART_API_HOST")
CHART_API_PORT = getenv("CHART_API_PORT")

COINGECKO_API_KEY = getenv("COINGECKO_API_KEY")

PRODUCTION = getenv("PRODUCTION")

MPC_SERVER_URL_1 = getenv("MPC_SERVER_URL_1")
MPC_SERVER_URL_2 = getenv("MPC_SERVER_URL_2")
MPC_SERVER_URL_3 = getenv("MPC_SERVER_URL_3")

BASE_CHAIN_ID = getenv("BASE_CHAIN_ID")
OPBNB_PROVIDER_RPC_URL = getenv("OPBNB_PROVIDER_RPC_URL")
OPBNB_USDT_TOKEN_ADDRESS = getenv("OPBNB_USDT_TOKEN_ADDRESS")
XP_TOKEN_CONTRACT_ADDRESS = getenv("XP_TOKEN_CONTRACT_ADDRESS")
XP_OWNER_ADDRESS = getenv("XP_OWNER_ADDRESS")
XP_OWNER_PRIVATE_KEY = getenv("XP_OWNER_PRIVATE_KEY")

