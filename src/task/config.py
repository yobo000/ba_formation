# -*- coding: utf-8 -*-

N = 3000  # 10000
# edge pramas 3
M = 3  # 3
# growth node number 50
M_0 = 30

PRECISION = 0.00001 # random_sample in (0,1)
INIT_SIZE = 300
# perferential attachment tolerant
THERSHOLD = 0.3
TOLERANT = 0.3
# deffuant tolerant
DEFFUANT_COEFF = 0.5

# GCP api
PROJECT = "**YOUR-PROJECT-NAME**"
METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
SERVICE_ACCOUNT = 'default'