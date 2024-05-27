#!/bin/bash
prefect init --recipe local && \
prefect work-pool create main_pool --type process --set-as-default
prefect --no-prompt deploy ./main.py:main_flow -n 'MLops' -p main_pool && \
prefect server start &
prefect worker start -t process -p main_pool