# Базовый образ с поддержкой CUDA 11.2 и Ubuntu 20.04
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Обновляем пакеты и устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip до последней версии
RUN python3 -m pip install --upgrade pip

# Установка PyTorch с поддержкой CUDA
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Копируем файл с зависимостями
COPY ./requirements.txt /tmp/requirements.txt

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Копируем все содержимое проекта в контейнер
COPY ./ /nas

# Устанавливаем рабочую директорию
WORKDIR /nas

# Устанавливаем пользовательскую команду по умолчанию
ENTRYPOINT ["sleep", "999999999999"]
