FROM python:3.6-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /app

EXPOSE 8080

# Run app.py when the container launches
# ENTRYPOINT ["python"]
CMD ["python","app.py"]
