FROM python:3.12.12-slim

ARG VENUS_FACTORY_CUDA_VARIANT=cpu

WORKDIR /app

COPY . .

RUN python .install/install_system.py --type ${VENUS_FACTORY_CUDA_VARIANT}

EXPOSE 7860

CMD ["python", "-u", "./src/webui.py"]