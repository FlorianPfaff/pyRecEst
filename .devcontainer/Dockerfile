FROM python:3.11

WORKDIR /workspace

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

COPY ./pyproject.toml /workspace/pyproject.toml
COPY ./poetry.lock* /workspace/

# Install dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi
